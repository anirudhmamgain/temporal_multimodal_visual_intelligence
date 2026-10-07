# models

Store trained checkpoints here.

The main temporal model is a single-layer GRU with hidden size 128 followed by a 128→64→2 classifier.
The baseline is a mean-pooled 768-dimensional representation followed by a small MLP.
