# Retrieval evaluation v1

`trusted_retrieval_v1.json` has the required deterministic shape: 120 atomic
cases (96 answerable and 24 negatives), eight evenly represented sources, an
80/40 development/held-out split, and 24 multi-claim scenarios split 16/8.

The asset is a review bootstrap. Claims, relevant chunk spans, and pooled top-10
judgments remain marked pending. `retrieval_eval_runner.py` validates the shape
but refuses benchmark or held-out execution until every review record is
complete. This prevents generated placeholders from being presented as human
judgments.

