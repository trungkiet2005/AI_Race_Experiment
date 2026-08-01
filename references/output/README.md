# Compiled and review output

`references/output/` is the tracked review surface for generated artifacts.
Local build scratch under the repository-root `output/` directory is ignored.

Current structure:

- `references/output/pdf/`
  - Compiled manuscript/deck PDFs (`.pdf` files are tracked).
  - LaTeX auxiliary artifacts (`.aux`, `.log`, etc.) are regenerated locally and ignored.
- `references/output/assets/`
  - A centralized folder for images and PDFs collected for quick review / sharing.

## Centralized asset snapshot

- [`references/output/assets/images/tracked/`](./assets/images/tracked): project PNG figures
  currently tracked in-repo (paper and web visuals).
- [`references/output/assets/images/source/`](./assets/images/source): project PDFs and source
  visuals mirrored for one-stop review.

External-paper artifacts are stored canonically in [`references/papers/`](../references/papers/)
instead of being duplicated under `output/`.
