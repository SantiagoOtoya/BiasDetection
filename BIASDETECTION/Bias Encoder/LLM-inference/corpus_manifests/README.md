# Corpus manifests

`initial-candidate-*.json` enumerates the required 40 source families and is
content-hash-addressed. It is a candidate, not an approved current manifest:
source versions, extraction hashes, expected chunk counts, and document-level
rights reviews are intentionally pending.

`manifest-approve --apply` writes immutable `manifest-<sha256>.json` files and a
hashed `current.json` pointer only after every document is staged and approved.
No current pointer is checked in because no such review has occurred.

