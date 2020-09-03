## Marge - Pre-training via Paraphrasing (wip)

Implementation of <a href="https://arxiv.org/abs/2006.15020">Marge</a>, Pre-training via Paraphrasing, in Pytorch

## Install

```bash
$ pip install marge-pytorch
```

## Usage

```python
import torch
from torch.utils.data import DataLoader
from marge_pytorch import Marge, TrainingWrapper

# your documents have already been tokenized

documents = torch.randint(0, 20000, (10000, 1024))
masks = torch.ones_like(documents).bool()

# instantiate model

model = Marge(
    dim = 512,
    num_tokens = 20000,
    max_seq_len = 1024,
    enc_depth = 12,
    enc_heads = 8,
    enc_ff_mult = 4,
    dec_depth = 12,
    dec_heads = 8,
    dec_ff_mult = 16
)

# wrap your model and your documents

trainer = TrainingWrapper(model, documents, masks = masks)

# instantiate dataloader

dl = DataLoader(trainer.dataset, batch_size=16)

# now you can train, and use the reindex method on the training wrapper at appropriate intervals

for ind, ids in enumerate(dl):
    loss = trainer(ids)
    loss.backward()
    # optimizer step and all that

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
