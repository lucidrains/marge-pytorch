import torch
import faiss
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange

from marge_pytorch.autoregressive_wrapper import AutoregressiveWrapper

# helpers

def identity(x, *args, **kwargs):
    return x

# helper classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, causal = True):
        super().__init__()
        self.scale = dim ** -0.5
        self.heads = heads
        self.causal = causal
        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        _, n, _, h, device = *x.shape, self.heads, x.device
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', h = h, qkv = 3)
        dots = einsum('bhid,bhjd->bhij', q, k) * self.scale

        if self.causal:
            mask = torch.ones(n, n, device=device).triu_(1).bool()
            dots.masked_fill_(mask, float('-inf'))

        attn = dots.softmax(dim=-1)
        out = einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class CrossAttention(nn.Module):
    def __init__(self, dim, heads = 8):
        super().__init__()
        self.scale = dim ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(dim, dim, bias = False)
        self.to_kv = nn.Linear(dim, dim * 2, bias = False)
        self.beta = nn.Parameter(torch.tensor(1.))
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, context, doc_similarities):
        _, n, _, h, device = *x.shape, self.heads, x.device

        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)

        context_len = context.shape[2]
        context = rearrange(context, 'b m n d -> b (m n) d')

        doc_similarities = doc_similarities.unsqueeze(-1).expand(-1, -1, context_len)
        doc_similarities = rearrange(doc_similarities, 'b m n -> b (m n)')
        doc_similarities = doc_similarities[:, None, None, :] * self.beta


        kv = self.to_kv(context)
        k, v = rearrange(kv, 'b n (kv h d) -> kv b h n d', h = h, kv = 2)

        dots = einsum('bhid,bhjd->bhij', q, k) * self.scale
        dots = dots + doc_similarities

        attn = dots.softmax(dim=-1)
        out = einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Encoder(nn.Module):
    def __init__(self, dim, depth, retrieval_encoder_depth = 4, heads = 8):
        super().__init__()
        assert depth > retrieval_encoder_depth, f'Depth must be at least the depth set for the retrieval encoder ({retrieval_encoder_depth})'

        block = lambda: nn.Sequential(
            Residual(Attention(dim)),
            Residual(FeedForward(dim))
        )

        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.encoder_head = nn.Sequential(*[block() for _ in range(retrieval_encoder_depth)])
        self.encoder_tail = nn.Sequential(*[block() for _ in range(depth - retrieval_encoder_depth)])

    def forward(self, x, fetch_document_embed = False):
        b, _, _ = x.shape
        cls_token = self.cls.expand(b, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        x_head = self.encoder_head(x)
        cls_tokens = x_head[:, 0]

        if fetch_document_embed:
            return cls_tokens

        return self.encoder_tail(x), cls_tokens

class Decoder(nn.Module):
    def __init__(self, dim, depth, head_depth = 4, heads = 8):
        super().__init__()
        block = lambda: nn.Sequential(
            Residual(Attention(dim, causal = True)),
            Residual(FeedForward(dim)),
        )

        self.decoder_head = nn.Sequential(*[block() for _ in range(head_depth)])

        self.decoder_tail = nn.ModuleList([])
        for _ in range(depth - head_depth):
            self.decoder_tail.append(nn.ModuleList([
                Residual(CrossAttention(dim)),
                Residual(FeedForward(dim))
            ]))

    def forward(self, x, context, doc_similarities):
        x_head = self.decoder_head(x)

        for attn, ff in self.decoder_tail:
            x = attn(x, context, doc_similarities)
            x = ff(x)

        return x

class TransformerWrapper(nn.Module):
    def __init__(self, num_tokens, dim, max_seq_len, layers, return_logits = False):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.max_seq_len = max_seq_len

        self.layers = layers
        self.to_logits = nn.Linear(dim, num_tokens) if return_logits else identity

    def forward(self, x, *args, **kwargs):
        b, n, device = *x.shape, x.device

        x = self.token_emb(x)
        x += self.pos_emb(torch.arange(n, device=device))

        x = self.layers(x, *args, **kwargs)
        return self.to_logits(x)

class Marge(nn.Module):
    def __init__(self, dim, num_tokens = 20000, max_seq_len = 1024, encoder_depth = 12, decoder_depth = 12):
        super().__init__()
        self.encoder = TransformerWrapper(num_tokens, dim, max_seq_len, Encoder(dim, depth = encoder_depth))
        self.decoder = AutoregressiveWrapper(TransformerWrapper(num_tokens, dim, max_seq_len, Decoder(dim, depth = decoder_depth), return_logits = True))

    def forward(self, evidence, target):
        all_docs = torch.cat((evidence, target.unsqueeze(1)), dim=1)
        num_docs = all_docs.shape[1]
        all_docs = rearrange(all_docs, 'b m n -> (b m) n')

        encodings, doc_embeds = self.encoder(all_docs)
        encodings = rearrange(encodings, '(b m) n d -> b m n d', m = num_docs)
        doc_embeds = rearrange(doc_embeds, '(b m) d -> b m d', m = num_docs)

        evidence_encodings = encodings[:, :-1]

        doc_embeds = F.normalize(doc_embeds, dim=-1)
        evidence_embeds, target_embeds = doc_embeds[:, :-1], doc_embeds [:, -1]
        similarities = einsum('bmd,bd->bm', evidence_embeds, target_embeds)

        return self.decoder(target, evidence_encodings, similarities)
