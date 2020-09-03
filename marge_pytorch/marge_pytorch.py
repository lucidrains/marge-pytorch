import faiss
import torch
from torch.utils.data import Dataset
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange

from marge_pytorch.autoregressive_wrapper import AutoregressiveWrapper

# helpers

def identity(x, *args, **kwargs):
    return x

# helper classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

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

class SelfAttention(nn.Module):
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
        self.beta = nn.Parameter(torch.tensor(1.), requires_grad=True)
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
            Residual(PreNorm(dim, SelfAttention(dim))),
            Residual(PreNorm(dim, FeedForward(dim)))
        )

        self.cls = nn.Parameter(torch.zeros(1, 1, dim), requires_grad=True)
        self.encoder_head = nn.Sequential(*[block() for _ in range(retrieval_encoder_depth)])
        self.encoder_tail = nn.Sequential(*[block() for _ in range(depth - retrieval_encoder_depth)])

    def forward(self, x, return_embed_only = False):
        b, _, _ = x.shape
        cls_token = self.cls.expand(b, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        x = self.encoder_head(x)
        cls_tokens = x[:, 0]

        if return_embed_only:
            return cls_tokens

        return self.encoder_tail(x), cls_tokens

class Decoder(nn.Module):
    def __init__(self, dim, depth, head_depth = 4, heads = 8):
        super().__init__()
        block = lambda: nn.Sequential(
            Residual(PreNorm(dim, SelfAttention(dim, causal = True))),
            Residual(PreNorm(dim, FeedForward(dim))),
        )

        self.decoder_head = nn.Sequential(*[block() for _ in range(head_depth)])

        self.decoder_tail = nn.ModuleList([])
        for _ in range(depth - head_depth):
            self.decoder_tail.append(nn.ModuleList([
                Residual(PreNorm(dim, CrossAttention(dim))),
                Residual(PreNorm(dim, FeedForward(dim)))
            ]))

    def forward(self, x, context, doc_similarities):
        x = self.decoder_head(x)

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
        self.dim = dim

        self.encoder = TransformerWrapper(num_tokens, dim, max_seq_len, Encoder(dim, depth = encoder_depth))
        self.decoder = AutoregressiveWrapper(TransformerWrapper(num_tokens, dim, max_seq_len, Decoder(dim, depth = decoder_depth), return_logits = True))

    def get_embeds(self, documents, batch_size = 16):
        embeds = []
        for batch in documents.split(batch_size):
            embed = self.encoder(batch, return_embed_only = True)
            embeds.append(embed)
        embeds = torch.cat(embeds)
        return F.normalize(embeds, dim=-1)

    def forward(self, evidence, target, target_embeds):
        num_evidences = evidence.shape[1]
        evidence = rearrange(evidence, 'b m n -> (b m) n')

        encodings, evidence_embeds = self.encoder(evidence)
        encodings = rearrange(encodings, '(b m) n d -> b m n d', m = num_evidences)
        evidence_embeds = rearrange(evidence_embeds, '(b m) d -> b m d', m = num_evidences)

        similarities = einsum('bmd,bd->bm', evidence_embeds, target_embeds)
        return self.decoder(target, encodings, similarities)

# training related classes

class DocumentDataset(Dataset):
    def __init__(self, documents):
        super().__init__()
        self.documents = documents

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, ind):
        return torch.tensor(ind).long()

def remove_target_from_evidence(evidence_ids, target_ids):
    b, n = evidence_ids.shape

    match_mask = evidence_ids == target_ids[:, None]
    rows_without_matches = (match_mask.sum(dim=-1) == 0)[:, None]
    remove_mask = torch.cat((torch.zeros(b, n - 1).bool(), rows_without_matches), dim=1)

    mask = match_mask + remove_mask
    filtered_ids = evidence_ids.masked_select(~mask)
    return filtered_ids.reshape(b, n - 1)

class TrainingWrapper(nn.Module):
    def __init__(self, model, documents, num_evidence = 4):
        super().__init__()
        self.dim = model.dim
        self.num_evidence = num_evidence

        self.model = model
        self.documents = documents

        self.index = None
        self.reindex()

        self.dataset = DocumentDataset(documents)

    def get_dataset(self):
        return self.dataset

    @torch.no_grad()
    def reindex(self):
        if self.index is not None:
            self.index.reset()
        else:
            self.index = faiss.IndexFlatL2(self.dim)
            self.index = faiss.index_cpu_to_all_gpus(self.index)

        embeds = self.model.get_embeds(self.documents)
        self.index.add(embeds.numpy())

    def forward(self, target_ids):
        targets = self.documents[target_ids]
        target_embeds = self.model.get_embeds(targets)
        dists, evidence_ids = self.index.search(target_embeds.detach().numpy(), k = self.num_evidence + 1)
        evidence_ids = torch.tensor(evidence_ids).long()
        evidence_ids = remove_target_from_evidence(evidence_ids, target_ids)
        evidences = self.documents[evidence_ids]
        loss = self.model(evidences, targets, target_embeds)
        return loss
