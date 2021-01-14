import faiss
import math
import numpy as np
from tqdm import tqdm
from einops import rearrange, repeat
from functools import partial
from contextlib import contextmanager

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, einsum
import torch.nn.functional as F

from marge_pytorch.autoregressive_wrapper import AutoregressiveWrapper

# helpers

def identity(x, *args, **kwargs):
    return x

def exists(x):
    return x is not None

def default(x, d):
    return x if exists(x) else d

def chunk(chunk_size, l):
    for lo in range(0, l, chunk_size):
        hi = min(l, lo + chunk_size)
        yield slice(lo, hi)

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

@contextmanager
def memmap(*args, **kwargs):
    pointer = np.memmap(*args, **kwargs)
    yield pointer
    del pointer

# attention distillation loss

def distill_attn_loss(evi_dots, doc_similarities, mask  = None, eps = 1e-5):
    evi_dots = rearrange(evi_dots, 'b l h i n j -> b (l h i) n j')

    if exists(mask):
        mask = rearrange(mask, 'b n j -> b () n j')
        evi_dots.masked_fill_(~mask, 0.)
        denom = mask.expand_as(evi_dots).sum(dim = (1, -1))
        evi_dots_mean = evi_dots.sum(dim = (1, -1)) / (denom + eps)
    else:
        evi_dots_mean = evi_dots.mean(dim = (1, -1))

    normed_evi_dots = evi_dots_mean.softmax(dim = -1)
    normed_evi_dots.detach_()

    doc_similarities = doc_similarities.softmax(dim = -1).log()
    loss = F.kl_div(doc_similarities, normed_evi_dots, reduction = 'batchmean')
    return loss

# helper classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return F.gelu(gates) * x

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        # to keep the number of parameters / computation constant with respect to non-GLU variant
        mult = int(mult / 3 * 2)

        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class SelfAttention(nn.Module):
    def __init__(self, dim, heads = 8, causal = True, dropout = 0.):
        super().__init__()
        self.scale = dim ** -0.5
        self.heads = heads
        self.causal = causal
        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask = None):
        _, n, _, h, device = *x.shape, self.heads, x.device
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', h = h, qkv = 3)
        dots = einsum('bhid,bhjd->bhij', q, k) * self.scale

        mask_value = max_neg_value(dots)

        if exists(mask):
            mask = mask[:, None, :, None] * mask[:, None, None, :]
            dots.masked_fill_(~mask, mask_value)
            del mask

        if self.causal:
            causal_mask = torch.ones(n, n, device=device).triu_(1).bool()
            dots.masked_fill_(causal_mask, mask_value)
            del causal_mask

        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class CrossAttention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.):
        super().__init__()
        self.scale = dim ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(dim, dim, bias = False)
        self.to_kv = nn.Linear(dim, dim * 2, bias = False)
        self.beta = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, doc_similarities, mask = None, context_mask = None):
        b, n, _, h, device = *x.shape, self.heads, x.device

        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)

        context_len = context.shape[2]
        context = rearrange(context, 'b m n d -> b (m n) d')
        context_mask = rearrange(context_mask, 'b m n -> b (m n)') if exists(context_mask) else None

        doc_similarities = repeat(doc_similarities, 'b m -> b m n', n=context_len)
        doc_similarities = rearrange(doc_similarities, 'b m n -> b (m n)')
        doc_similarities = doc_similarities[:, None, None, :] * self.beta

        kv = self.to_kv(context)
        k, v = rearrange(kv, 'b n (kv h d) -> kv b h n d', h = h, kv = 2)

        dots = einsum('bhid,bhjd->bhij', q, k) * self.scale
        pre_attn_dots = dots

        dots = dots + doc_similarities

        if any(map(exists, (mask, context_mask))):
            if not exists(mask):
                mask = torch.full((b, n), True, dtype=torch.bool, device=device)

            if not exists(context_mask):
                context_mask = torch.full(context.shape[:2], True, dtype=torch.bool, device=device)

            cross_mask = mask[:, None, :, None] * context_mask[:, None, None, :]
            mask_value = max_neg_value(dots)
            dots.masked_fill_(~cross_mask, mask_value)
            del cross_mask

        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out, pre_attn_dots

class Encoder(nn.Module):
    def __init__(self, dim, depth, retrieval_depth = 4, heads = 8, ff_mult = 4, attn_dropout = 0., ff_dropout = 0.):
        super().__init__()
        assert depth > retrieval_depth, f'Depth must be at least the depth set for the retrieval encoder ({retrieval_depth})'

        block = lambda: nn.ModuleList([
            PreNorm(dim, SelfAttention(dim, causal=False, dropout = attn_dropout)),
            PreNorm(dim, FeedForward(dim, mult = ff_mult))
        ])

        self.cls = nn.Parameter(torch.zeros(1, dim), requires_grad=True)
        self.encoder_head = nn.ModuleList([])
        self.encoder_tail = nn.ModuleList([])

        for _ in range(retrieval_depth):
            self.encoder_head.append(block())

        for _ in range(depth - retrieval_depth):
            self.encoder_tail.append(block())

    def forward(self, x, src_mask = None, return_embed_only = False):
        b, _, _ = x.shape

        # append cls token
        cls_token = repeat(self.cls, 'n d -> b n d', b=b)
        x = torch.cat((cls_token, x), dim=1)
        src_mask = F.pad(src_mask, (1, 0), value=True) if exists(src_mask) else None

        for attn, ff in self.encoder_head:
            x = attn(x, mask = src_mask) + x
            x = ff(x) + x

        cls_tokens = x[:, 0]
        
        if return_embed_only:
            return cls_tokens, None

        for attn, ff in self.encoder_tail:
            x = attn(x, mask = src_mask) + x
            x = ff(x) + x

        return x[:, 1:], cls_tokens

class Decoder(nn.Module):
    def __init__(self, dim, depth, head_depth = 4, heads = 8, ff_mult = 4, attn_dropout = 0., ff_dropout = 0.):
        super().__init__()
        self.decoder_head = nn.ModuleList([])
        self.decoder_tail = nn.ModuleList([])

        for _ in range(head_depth):
            self.decoder_head.append(nn.ModuleList([
                PreNorm(dim, SelfAttention(dim, causal = True, dropout = attn_dropout)),
                PreNorm(dim, FeedForward(dim))
            ]))

        for _ in range(depth - head_depth):
            self.decoder_tail.append(nn.ModuleList([
                PreNorm(dim, SelfAttention(dim, causal = True, dropout = attn_dropout)),
                PreNorm(dim, FeedForward(dim)),
                PreNorm(dim, CrossAttention(dim, dropout = attn_dropout)),
                PreNorm(dim, FeedForward(dim, mult = ff_mult))
            ]))

    def forward(self, x, *, context, similarities, src_mask = None, context_mask = None):
        for self_attn, self_ff in self.decoder_head:
            x = self_attn(x, mask = src_mask) + x
            x = self_ff(x) + x

        cross_pre_attns = []

        for self_attn, self_ff, cross_attn, cross_ff in self.decoder_tail:
            x = self_attn(x, mask = src_mask) + x
            x = self_ff(x) + x

            x_out, attn = cross_attn(x, context, similarities, mask = src_mask, context_mask = context_mask)
            x = x_out + x

            x = cross_ff(x) + x

            cross_pre_attns.append(attn)

        return x, cross_pre_attns

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
        assert n <= self.max_seq_len, f'your sequence length {n} needs to be less than or equal to the max sequence length {self.max_seq_len}'

        x = self.token_emb(x)
        x += self.pos_emb(torch.arange(n, device=device))

        x, *out = self.layers(x, *args, **kwargs)
        return (self.to_logits(x), *out)

class Marge(nn.Module):
    def __init__(
        self,
        dim,
        num_tokens = 20000,
        max_seq_len = 1024,
        enc_depth = 12,
        enc_retrieval_depth = 4,
        enc_heads = 8,
        enc_ff_mult = 4,
        enc_attn_dropout = 0.,
        enc_ff_dropout = 0.,
        dec_depth = 12,
        dec_heads = 8,
        dec_ff_mult = 16,
        dec_attn_dropout = 0.,
        dec_ff_dropout = 0.,
        distill_attn = False,
        distill_loss_coef = 1.
    ):
        super().__init__()
        self.dim = dim

        self.encoder = TransformerWrapper(num_tokens, dim, max_seq_len, Encoder(dim, depth = enc_depth, retrieval_depth = enc_retrieval_depth, heads = enc_heads, ff_mult = enc_ff_mult, attn_dropout = enc_attn_dropout, ff_dropout = enc_ff_dropout))
        self.decoder = TransformerWrapper(num_tokens, dim, max_seq_len, Decoder(dim, depth = dec_depth, heads = dec_heads, ff_mult = dec_ff_mult, attn_dropout = dec_attn_dropout, ff_dropout = dec_ff_dropout), return_logits = True)
        self.encoder.token_emb = self.decoder.token_emb

        self.decoder = AutoregressiveWrapper(self.decoder)

        # experimental attn distillation settings
        self.distill_attn = distill_attn
        self.distill_loss_coef = distill_loss_coef

    def get_embeds(self, documents, batch_size = 16, masks = None):
        embeds = []

        batched_documents = documents.split(batch_size)
        batched_masks = masks.split(batch_size) if exists(masks) else ([None] * len(batched_documents))

        for docs, mask in zip(batched_documents, batched_masks):
            embed, *_ = self.encoder(docs, src_mask = mask, return_embed_only = True)
            embeds.append(embed)

        embeds = torch.cat(embeds)
        return F.normalize(embeds, dim=-1)

    @torch.no_grad()
    def generate(self, prime, seq_len, evidence, mask = None, similarities = None):
        b, num_evidences, *_ = evidence.shape
        evidence = rearrange(evidence, 'b m n -> (b m) n')
        enc_src_mask = rearrange(mask, 'b m n -> (b m) n') if exists(mask) else None

        encodings, evidence_embeds = self.encoder(evidence, src_mask = enc_src_mask)
        encodings = rearrange(encodings, '(b m) n d -> b m n d', m = num_evidences)

        similarities = similarities if exists(similarities) else torch.ones((b, num_evidences)).float().cuda()
        context_mask = F.pad(mask, (1, 0), value = True) if exists(mask) else None
        return self.decoder.generate(prime, seq_len, context = encodings, similarities = similarities, context_mask = context_mask)

    def forward(self, evidence, target, target_embeds, src_mask = None, tgt_mask = None):
        num_evidences = evidence.shape[1]
        evidence = rearrange(evidence, 'b m n -> (b m) n')
        enc_src_mask = rearrange(src_mask, 'b m n -> (b m) n') if exists(src_mask) else None
        encodings, evidence_embeds = self.encoder(evidence, src_mask = enc_src_mask)
        encodings = rearrange(encodings, '(b m) n d -> b m n d', m = num_evidences)
        evidence_embeds = rearrange(evidence_embeds, '(b m) d -> b m d', m = num_evidences)

        similarities = einsum('bmd,bd->bm', evidence_embeds, target_embeds)

        dec_src_mask = tgt_mask[:, :-1] if exists(tgt_mask) else None
        loss, cross_attns = self.decoder(target, context = encodings, similarities = similarities, src_mask = dec_src_mask, context_mask = src_mask)

        if self.distill_attn:
            cross_attns = torch.stack(cross_attns, dim = 1)
            cross_attns = rearrange(cross_attns, 'b l h i (n j) -> b l h i n j', n = num_evidences)
            distill_loss = distill_attn_loss(cross_attns, similarities, mask = src_mask)
            aux_loss = self.distill_loss_coef * distill_loss
            loss = loss + aux_loss

        return loss

# training related classes

def remove_target_from_evidence(evidence_ids, target_ids):
    b, n = evidence_ids.shape

    match_mask = evidence_ids == target_ids[:, None]
    rows_without_matches = (match_mask.sum(axis=-1) == 0)[:, None]
    remove_mask = np.concatenate((np.full((b, n - 1), False), rows_without_matches), axis=1)

    mask = match_mask + remove_mask
    filtered_ids = evidence_ids[~mask]
    return filtered_ids.reshape(b, n - 1)

class DocumentDataset(Dataset):
    def __init__(self, num_docs, doc_seq_len, num_evidences, documents_path, masks_path, num_targets, target_seq_len, target_path, target_masks_path):
        super().__init__()
        self.shape = (num_docs, doc_seq_len)
        self.target_shape = (num_targets, target_seq_len)
        self.knn_shape = (num_targets, num_evidences)
        self.documents = np.memmap(documents_path, dtype=np.int32, shape=self.shape)
        self.targets = np.memmap(target_path, dtype=np.int32, shape=self.target_shape)
        self.masks = np.memmap(masks_path, dtype=np.bool, shape=self.shape) if exists(masks_path) else None
        self.target_masks = np.memmap(target_masks_path, dtype=np.bool, shape=self.target_shape) if exists(target_masks_path) else None
        self.knn = None

    def set_knn_path(self, path):
        if exists(self.knn):
            del self.knn
        self.knn = np.memmap(path, dtype=np.int32, shape=self.knn_shape)

    def __len__(self):
        return self.target_shape[0]

    def __getitem__(self, ind):
        assert exists(self.knn), 'The memmap path to the generated k nearest neighbors for evidences must be set for the dataset'

        target_data = torch.from_numpy(self.targets[ind, :]).long()
        target_masks = torch.from_numpy(self.target_masks[ind, :]) if exists(self.target_masks) else torch.ones_like(target_data).bool()

        evidence_ids = self.knn[ind, :]
        evidence_data = torch.from_numpy(self.documents[evidence_ids, :]).long()
        evidence_masks = torch.from_numpy(self.masks[evidence_ids, :]) if exists(self.masks) else torch.ones_like(evidence_data).bool()
        return target_data.cuda(), target_masks.cuda(), evidence_data.cuda(), evidence_masks.cuda()

class FaissANN():
    def __init__(
        self,
        dim,
        num_documents,
        num_subvectors = 16,
        hnsw_m = 32,
        nbits = 8
    ):
        super().__init__()
        nlist = math.floor(math.sqrt(num_documents))
        quantizer = faiss.IndexHNSWFlat(dim, hnsw_m)
        index = faiss.IndexIVFPQ(quantizer, dim, nlist, num_subvectors, nbits)
        self.index = faiss.index_cpu_to_all_gpus(index)
        self.num_training = max(nlist * 10, 256)

    def reset(self):
        return self.index.reset()

    def train(self, x):
        return self.index.train(x)

    def add(self, x):
        return self.index.add(x)

    def search(self, x, topk, nprobe=8):
        self.index.nprobe = nprobe
        return self.index.search(x, k=topk)

class TrainingWrapper(nn.Module):
    def __init__(
        self,
        model,
        *,
        num_documents,
        doc_seq_len,
        documents_memmap_path,
        masks_memmap_path = None,
        num_targets = None,
        target_seq_len = None,
        target_memmap_path = None,
        target_masks_memmap_path = None,
        num_evidence = 4,
        reindex_batch_size = 4,
        use_faiss_ann = False
    ):
        super().__init__()
        self.dim = model.dim
        self.num_evidence = num_evidence

        self.model = model.cuda()
        self.num_docs = num_documents

        num_targets = default(num_targets, num_documents)
        self.num_targets = num_targets

        self.doc_shape = (num_documents, doc_seq_len)

        self.documents_path = documents_memmap_path
        self.separate_target_and_evidence = exists(target_memmap_path)

        if self.separate_target_and_evidence:
            assert exists(num_targets), 'number of target documents must be defined if target document set is different than evidence document set'
            assert exists(target_seq_len), 'target sequence length must be specified'
        else:
            target_memmap_path = default(target_memmap_path, documents_memmap_path)
            target_masks_memmap_path = default(target_masks_memmap_path, masks_memmap_path)
            target_seq_len = default(target_seq_len, doc_seq_len)

        self.target_shape = (num_targets, target_seq_len)
        self.target_path = target_memmap_path
        self.knn_path = f'{self.documents_path}.knn'

        self.use_faiss_ann = use_faiss_ann
        if use_faiss_ann:
            self.index = FaissANN(self.dim, self.num_docs)
        else:
            index = faiss.IndexFlatL2(self.dim)
            self.index = faiss.index_cpu_to_all_gpus(index)

        self.reindex_batch_size = reindex_batch_size
        self.reindex()

        self.dataset = DocumentDataset(
            num_documents,
            doc_seq_len,
            num_evidence,
            documents_memmap_path,
            masks_memmap_path,
            num_targets,
            target_seq_len,
            target_memmap_path,
            target_masks_memmap_path
        )

        self.dataset.set_knn_path(self.knn_path)

    def get_dataset(self):
        return self.dataset

    @torch.no_grad()
    def reindex(self):
        batch_size = self.reindex_batch_size

        def get_embeds(data):
            embeds = self.model.get_embeds(data, batch_size = batch_size)
            return embeds.detach().cpu().numpy()

        with memmap(self.documents_path, dtype=np.int32, shape=self.doc_shape) as (doc_pointer
            ), memmap(self.target_path, dtype=np.int32, shape=self.target_shape) as (target_pointer
            ), memmap(self.knn_path, dtype=np.int32, shape=(self.num_docs, self.num_evidence), mode='w+') as knn_writer:

            if self.use_faiss_ann:
                random_indices = np.random.permutation(self.num_docs)[:self.index.num_training]
                np_data = torch.from_numpy(doc_pointer[random_indices]).cuda().long()
                train_embeds = get_embeds(np_data)
                self.index.train(train_embeds)

            total_evidence_chunks = math.ceil(self.num_docs / batch_size)

            for data_slice in tqdm(chunk(batch_size, self.num_docs), total=total_evidence_chunks, desc='Adding embedding to indexes'):
                np_data = torch.from_numpy(doc_pointer[data_slice, :]).cuda().long()
                embeds = get_embeds(np_data)
                self.index.add(embeds)

            total_target_chunks = math.ceil(self.num_targets / batch_size)

            for data_slice in tqdm(chunk(batch_size, self.num_targets), total=total_target_chunks, desc='Fetching and storing nearest neighbors'):
                np_data = torch.from_numpy(target_pointer[data_slice, :]).cuda().long()

                embeds = get_embeds(np_data)
                fetch_num_evidences = self.num_evidence + (0 if self.separate_target_and_evidence else 1)
                _, evidence_ids = self.index.search(embeds, fetch_num_evidences)

                target_ids = np.arange(data_slice.start, data_slice.stop)

                if not self.separate_target_and_evidence:
                    evidence_ids = remove_target_from_evidence(evidence_ids, target_ids)

                knn_writer[data_slice, :] = evidence_ids

        self.index.reset()

        print('reindexing complete')

    def forward(self, data):
        targets, target_masks, evidences, evidence_masks = data
        target_embeds = self.model.get_embeds(targets, masks = target_masks)
        loss = self.model(evidences, targets, target_embeds, src_mask = evidence_masks, tgt_mask = target_masks)
        return loss
