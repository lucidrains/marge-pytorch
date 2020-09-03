import faiss
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, einsum
import torch.nn.functional as F

import numpy as np
from einops import rearrange

from marge_pytorch.autoregressive_wrapper import AutoregressiveWrapper

# helpers

def identity(x, *args, **kwargs):
    return x

def exists(x):
    return x is not None

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

    def forward(self, x, mask = None):
        _, n, _, h, device = *x.shape, self.heads, x.device
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', h = h, qkv = 3)
        dots = einsum('bhid,bhjd->bhij', q, k) * self.scale

        if exists(mask):
            mask = mask[:, None, :, None] * mask[:, None, None, :]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        if self.causal:
            causal_mask = torch.ones(n, n, device=device).triu_(1).bool()
            dots.masked_fill_(causal_mask, float('-inf'))
            del causal_mask

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

    def forward(self, x, context, doc_similarities, mask = None, context_mask = None):
        b, n, _, h, device = *x.shape, self.heads, x.device

        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)

        context_len = context.shape[2]
        context = rearrange(context, 'b m n d -> b (m n) d')
        context_mask = rearrange(context_mask, 'b m n -> b (m n)') if exists(context_mask) else None

        doc_similarities = doc_similarities.unsqueeze(-1).expand(-1, -1, context_len)
        doc_similarities = rearrange(doc_similarities, 'b m n -> b (m n)')
        doc_similarities = doc_similarities[:, None, None, :] * self.beta

        kv = self.to_kv(context)
        k, v = rearrange(kv, 'b n (kv h d) -> kv b h n d', h = h, kv = 2)

        dots = einsum('bhid,bhjd->bhij', q, k) * self.scale
        dots = dots + doc_similarities

        if any(map(exists, (mask, context_mask))):
            if not exists(mask):
                mask = torch.tensor((b, n), True, device=device)

            if not exists(context_mask):
                context_mask = torch.tensor((b, context_len), True, device=device)

            cross_mask = mask[:, None, :, None] * context_mask[:, None, None, :]
            dots.masked_fill_(~cross_mask, float('-inf'))
            del cross_mask

        attn = dots.softmax(dim=-1)
        out = einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Encoder(nn.Module):
    def __init__(self, dim, depth, retrieval_encoder_depth = 4, heads = 8, ff_mult = 4):
        super().__init__()
        assert depth > retrieval_encoder_depth, f'Depth must be at least the depth set for the retrieval encoder ({retrieval_encoder_depth})'

        block = lambda: nn.ModuleList([
            Residual(PreNorm(dim, SelfAttention(dim))),
            Residual(PreNorm(dim, FeedForward(dim, mult = ff_mult)))
        ])

        self.cls = nn.Parameter(torch.zeros(1, 1, dim), requires_grad=True)
        self.encoder_head = nn.ModuleList([])
        self.encoder_tail = nn.ModuleList([])

        for _ in range(retrieval_encoder_depth):
            self.encoder_head.append(block())

        for _ in range(depth - retrieval_encoder_depth):
            self.encoder_tail.append(block())

    def forward(self, x, src_mask = None, return_embed_only = False):
        b, _, _ = x.shape

        # append cls token
        cls_token = self.cls.expand(b, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        src_mask = F.pad(src_mask, (1, 0), value=True) if src_mask is not None else None

        for attn, ff in self.encoder_head:
            x = attn(x, mask = src_mask)
            x = ff(x)

        cls_tokens = x[:, 0]

        if return_embed_only:
            return cls_tokens

        for attn, ff in self.encoder_tail:
            x = attn(x, mask = src_mask)
            x = ff(x)

        return x, cls_tokens

class Decoder(nn.Module):
    def __init__(self, dim, depth, head_depth = 4, heads = 8, ff_mult = 4):
        super().__init__()
        self.decoder_head = nn.ModuleList([])
        self.decoder_tail = nn.ModuleList([])

        for _ in range(head_depth):
            self.decoder_head.append(nn.ModuleList([
                Residual(PreNorm(dim, SelfAttention(dim, causal = True))),
                Residual(PreNorm(dim, FeedForward(dim)))
            ]))

        for _ in range(depth - head_depth):
            self.decoder_tail.append(nn.ModuleList([
                Residual(PreNorm(dim, SelfAttention(dim, causal = True))),
                Residual(PreNorm(dim, FeedForward(dim))),
                Residual(PreNorm(dim, CrossAttention(dim))),
                Residual(PreNorm(dim, FeedForward(dim, mult = ff_mult)))
            ]))

    def forward(self, x, context, doc_similarities, src_mask = None, context_mask = None):
        for self_attn, self_ff in self.decoder_head:
            x = self_attn(x, mask = src_mask)
            x = self_ff(x)

        for self_attn, self_ff, cross_attn, cross_ff in self.decoder_tail:
            x = self_attn(x, mask = src_mask)
            x = self_ff(x)
            x = cross_attn(x, context, doc_similarities, mask = src_mask, context_mask = context_mask)
            x = cross_ff(x)

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
    def __init__(
        self,
        dim,
        num_tokens = 20000,
        max_seq_len = 1024,
        enc_depth = 12,
        enc_heads = 8,
        enc_ff_mult = 4,
        dec_depth = 12,
        dec_heads = 8,
        dec_ff_mult = 16
    ):
        super().__init__()
        self.dim = dim

        self.encoder = TransformerWrapper(num_tokens, dim, max_seq_len, Encoder(dim, depth = enc_depth, heads = enc_heads, ff_mult = enc_ff_mult))
        self.decoder = AutoregressiveWrapper(TransformerWrapper(num_tokens, dim, max_seq_len, Decoder(dim, depth = dec_depth, heads = dec_heads, ff_mult = dec_ff_mult), return_logits = True))

    def get_embeds(self, documents, batch_size = 16, masks = None):
        embeds = []

        batched_documents = documents.split(batch_size)
        batched_masks = masks.split(batch_size) if masks is not None else ([None] * len(batched_documents))

        for docs, mask in zip(batched_documents, batched_masks):
            embed = self.encoder(docs, src_mask = mask, return_embed_only = True)
            embeds.append(embed)

        embeds = torch.cat(embeds)
        return F.normalize(embeds, dim=-1)

    def forward(self, evidence, target, target_embeds, src_mask = None, tgt_mask = None):
        num_evidences = evidence.shape[1]
        evidence = rearrange(evidence, 'b m n -> (b m) n')
        enc_src_mask = rearrange(src_mask, 'b m n -> (b m) n') if src_mask is not None else None

        encodings, evidence_embeds = self.encoder(evidence, src_mask = enc_src_mask)
        encodings = rearrange(encodings, '(b m) n d -> b m n d', m = num_evidences)
        evidence_embeds = rearrange(evidence_embeds, '(b m) d -> b m d', m = num_evidences)

        similarities = einsum('bmd,bd->bm', evidence_embeds, target_embeds)
        dec_src_mask = F.pad(src_mask, (1, 0), value = True)
        return self.decoder(target, encodings, similarities, src_mask = tgt_mask[:,:-1], context_mask = dec_src_mask)

# training related classes

class DocumentDataset(Dataset):
    def __init__(self, num_docs, doc_seq_len, num_evidences, documents_path, masks_path):
        super().__init__()
        self.shape = (num_docs, doc_seq_len)
        self.knn_shape = (num_docs, num_evidences)
        self.documents = np.memmap(documents_path, dtype=np.int32, shape=self.shape)
        self.masks = np.memmap(masks_path, dtype=np.bool, shape=self.shape) if exists(masks_path) else None
        self.knn = None

    def set_knn_path(self, path):
        if exists(self.knn):
            del self.knn
        self.knn = np.memmap(path, dtype=np.int32, shape=self.knn_shape)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, ind):
        assert exists(self.knn), 'The memmap path to the generated k nearest neighbors for evidences must be set for the dataset'

        target_data = torch.from_numpy(self.documents[ind, :]).long()
        target_masks = torch.from_numpy(self.masks[ind, :]) if exists(self.masks) else torch.ones_like(target_data).bool()

        evidence_ids = self.knn[ind, :]
        evidence_data = torch.from_numpy(self.documents[evidence_ids, :]).long()
        evidence_masks = torch.from_numpy(self.masks[evidence_ids, :]) if exists(self.masks) else torch.ones_like(evidence_data).bool()

        return target_data.cuda(), target_masks.cuda(), evidence_data.cuda(), evidence_masks.cuda()

def remove_target_from_evidence(evidence_ids, target_ids):
    b, n = evidence_ids.shape

    match_mask = evidence_ids == target_ids[:, None]
    rows_without_matches = (match_mask.sum(axis=-1) == 0)[:, None]
    remove_mask = np.concatenate((np.full((b, n - 1), False), rows_without_matches), axis=1)

    mask = match_mask + remove_mask
    filtered_ids = evidence_ids[~mask]
    return filtered_ids.reshape(b, n - 1)

class TrainingWrapper(nn.Module):
    def __init__(
        self,
        model,
        *,
        num_documents,
        doc_seq_len,
        documents_memmap_path,
        masks_memmap_path = None,
        num_evidence = 4,
        reindex_batch_size = 4
    ):
        super().__init__()
        self.dim = model.dim
        self.num_evidence = num_evidence

        self.model = model.cuda()
        self.num_docs = num_documents
        self.doc_shape = (num_documents, doc_seq_len)
        self.documents_path = documents_memmap_path
        self.knn_path = f'{self.documents_path}.knn'

        self.index = None
        self.reindex_batch_size = reindex_batch_size
        self.reindex()

        self.dataset = DocumentDataset(
            num_documents,
            doc_seq_len,
            num_evidence,
            documents_memmap_path,
            masks_memmap_path
        )

        self.dataset.set_knn_path(self.knn_path)

    def get_dataset(self):
        return self.dataset

    @torch.no_grad()
    def reindex(self):
        batch_size = self.reindex_batch_size

        if exists(self.index):
            self.index.reset()
        else:
            self.index = faiss.IndexFlatL2(self.dim)
            self.index = faiss.index_cpu_to_all_gpus(self.index)

        doc_pointer = np.memmap(self.documents_path, dtype=np.int32, shape=self.doc_shape)

        for ind in range(0, self.num_docs, batch_size):
            data_slice = slice(ind, min(ind + batch_size, self.num_docs))
            np_data = torch.from_numpy(doc_pointer[data_slice, :]).cuda().long()
            embeds = self.model.get_embeds(np_data, batch_size = batch_size)
            self.index.add(embeds.detach().cpu().numpy())


        knn_writer = np.memmap(self.knn_path, dtype=np.int32, shape=(self.num_docs, self.num_evidence), mode='w+')

        for ind in range(0, self.num_docs, batch_size):
            lo, hi = ind, min(ind + batch_size, self.num_docs)
            data_slice = slice(lo, hi)
            np_data = torch.from_numpy(doc_pointer[data_slice, :]).cuda().long()

            embeds = self.model.get_embeds(np_data, batch_size = batch_size)
            _, evidence_ids = self.index.search(embeds.detach().cpu().numpy(), k = self.num_evidence + 1)

            target_ids = np.arange(lo, hi)
            evidence_ids = remove_target_from_evidence(evidence_ids, target_ids)

            knn_writer[data_slice, :] = evidence_ids

        del doc_pointer
        del knn_writer

    def forward(self, data):
        targets, target_masks, evidences, evidence_masks = data
        target_embeds = self.model.get_embeds(targets, masks = target_masks)
        loss = self.model(evidences, targets, target_embeds, src_mask = evidence_masks, tgt_mask = target_masks)
        return loss
