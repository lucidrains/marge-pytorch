<img src="./marge.png" width="600px"></img>

## Marge - Pre-training via Paraphrasing (wip)

Implementation of <a href="https://arxiv.org/abs/2006.15020">Marge</a>, Pre-training via Paraphrasing, in Pytorch. It is an alternative to masked language modeling pretraining, where an encoder / decoder attention network learns to reconstruct a target document from a collection of evidence documents.

## Install

```bash
$ pip install marge-pytorch
```

## Usage

```python
import torch
import numpy as np
from torch.utils.data import DataLoader

from marge_pytorch import Marge, TrainingWrapper

# your documents must be tokenized and stored as memmap in the shape (num documents, seq length)

# mock constants
NUM_DOCS = 10000
SEQ_LEN = 1024

# generate mock training data
f = np.memmap('./train.dat', dtype=np.int32, mode='w+', shape=(NUM_DOCS, SEQ_LEN))
f[:, :] = np.random.rand(NUM_DOCS, SEQ_LEN)
del f

# generate mock masking data
f = np.memmap('./train.mask.dat', dtype=np.bool, mode='w+', shape=(NUM_DOCS, SEQ_LEN))
f[:, :] = np.full((NUM_DOCS, SEQ_LEN), True)
del f

# instantiate model

model = Marge(
    dim = 512,
    num_tokens = 20000,
    max_seq_len = SEQ_LEN,
    enc_depth = 12,
    enc_heads = 8,
    enc_ff_mult = 4,
    dec_depth = 12,
    dec_heads = 8,
    dec_ff_mult = 16
)

# wrap your model and your documents

trainer = TrainingWrapper(
    model,
    num_documents = NUM_DOCS,
    doc_seq_len = SEQ_LEN,
    num_evidence = 4,
    reindex_batch_size = 32,
    documents_memmap_path = './path/to/train.dat',
    masks_memmap_path = './path/to/train.mask.dat'
)

# instantiate dataloader

dl = DataLoader(trainer.dataset, batch_size=16)

# now you can train, and use the reindex method on the training wrapper at appropriate intervals

for ind, ids in enumerate(dl):
    loss = trainer(ids)
    loss.backward()
    # optimizer step and all that

    # reindex and precompute knn every 10000 steps, as in paper
    if ind % 10000 == 0:
        trainer.reindex()
```

## Citations

```bibtex
@misc{lewis2020pretraining,
    title={Pre-training via Paraphrasing},
    author={Mike Lewis and Marjan Ghazvininejad and Gargi Ghosh and Armen Aghajanyan and Sida Wang and Luke Zettlemoyer},
    year={2020},
    eprint={2006.15020},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
