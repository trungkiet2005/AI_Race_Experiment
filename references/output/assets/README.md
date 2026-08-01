# Output assets mirror

This folder stores a consolidated review set of visual artifacts.

`tracked/` contains project PNG assets that are already tracked in the repository.
`source/` mirrors project PDFs for convenient grouping.

Artifacts imported from external papers live only in `references/papers/`; they
are intentionally not mirrored here.

## Why this exists

- keep all visual assets in one place for sharing,
- avoid chasing many folders while presenting,
- provide a clean entry point for slides/papers/visual reports.

When you run a new experiment that emits new figures, just add them here as:

```bash
Copy-Item path\to\new\figure.png references/output/assets/images/tracked\
```

or extend this folder with a tiny metadata table in this file.
