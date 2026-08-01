# `output/` in this repo

`output/` is intentionally used for generated artifacts, not source code.

Current structure:

- `output/pdf/`
  - Compiled manuscript/deck PDFs (`.pdf` files are tracked).
  - LaTeX auxiliary artifacts (`.aux`, `.log`, etc.) are regenerated locally and ignored.
- `output/game_understanding/`
  - Raw simulator audit runs and local logs from experiments.
  - This folder is intentionally ignored by Git (`.gitignore`) to keep the repo clean.
- `output/assets/`
  - A centralized folder for images and PDFs collected for quick review / sharing.

If you want to keep this folder strictly minimal, we can remove the raw ignored
`game_understanding/` directory after each run (it is safe to delete because it is
not committed).

Quick clean command:

```bash
Remove-Item -Recurse -Force output/game_understanding
```

## Centralized asset snapshot

- [`output/assets/images/tracked/`](./assets/images/tracked): project PNG figures
  currently tracked in-repo (paper and web visuals).
- [`output/assets/images/source/`](./assets/images/source): project PDFs and source
  visuals mirrored for one-stop review.

External-paper artifacts are stored canonically in [`references/papers/`](../references/papers/)
instead of being duplicated under `output/`.
