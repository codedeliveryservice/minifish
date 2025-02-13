## minifish

NNUE chess engine that compresses to less than 64kb.

The implementation was based on:

- [Cfish](https://github.com/syzygy1/Cfish) - engine source code foundation
- [Obsidian](https://github.com/gab8192/Obsidian) - elegant NNUE reference code
- [bullet](https://github.com/jw1912/bullet) - for training NNUE
- [Primer](https://github.com/PGG106/Primer) - for creating .bullet format training data, modified to improve filtering

Training data:

- [source binpack training data](https://robotmoon.com/nnue-training-data/) - T77 unfiltered, T79 v6-dd, used as source data
- [final bullet training data](https://huggingface.co/datasets/linrock/bullet-training-data/tree/main) - filtered data used for training

Created for the [FIDE & Google Efficient Chess AI Challenge](https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge)
