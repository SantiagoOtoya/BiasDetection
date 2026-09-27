# SBERT Classification Architecture

The bias encoder fine-tunes `sentence-transformers/all-mpnet-base-v2` for
media-bias sentence classification. The model uses mean pooling over token
embeddings to produce a fixed-size sentence representation.

The training setup uses BABE media-bias labels and keeps the lower MPNet layers
frozen while fine-tuning the upper encoder layers. This preserves the base
model's general sentence-embedding behavior while adapting the final layers to
media-bias and opinion-language distinctions.

Two one-hidden-layer MLP classification heads are trained on top of the shared
SBERT encoder:

- a primary bias classification head
- an auxiliary opinion-label classification head

Both heads are optimized in the same forward and backward pass:

```text
total_loss = bias_loss + alpha * opinion_loss
```

The default training configuration uses `alpha=0.3`, dropout of `0.2`, and
AdamW weight decay of `0.01`.
