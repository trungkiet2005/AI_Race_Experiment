# Figure hub

This folder is the single place to browse, compare, and select project figures.

## Layout

- `paper/` contains the current PDF and raster assets referenced by the paper and research deck.
- `gallery/` contains the complete candidate gallery, including admitted, exploratory, selected, and redrawn views.
- `diagnostics/` contains hash-tracked copies of additional exports from analysis and result directories.
- `INDEX.md` is the machine-generated visual index.
- `manifest.json` records SHA256, source paths, and evidence status for every unique visual file.

The original files under `results/` and `analysis/` remain the provenance source for diagnostics. A copy in this hub does not change its evidence class. Check the relevant run manifest and admission report before promoting a diagnostic to the paper.

## Workflow

1. Generate or regenerate the result in its source analysis directory.
2. Run `python scripts/build_publication_figures.py` to rebuild the generated
   canonical main-paper and supplementary figures from checked-in sources.
   Figures 1, 2, 4, and 8 are author-supplied artwork; the builder validates
   their presence and refuses to overwrite them.
3. Run `python scripts/build_figure_gallery.py` from the repository root.
4. Inspect `INDEX.md` and the linked source artifact.
5. Update the manuscript reference only after the figure passes the paper's evidence and visual QA gates.

`scripts/build_publication.py --paper-only` runs the canonical figure builder
automatically before compiling both PDFs. The builder is the source of truth
for generated figures; the four protected artwork files are checked for
presence and provenance but are not regenerated. Exported PDF, PNG, and SVG
files are paper-ready derivatives, not independent analyses.

The supplementary subset is built by
[`scripts/build_supplementary_figures.py`](../scripts/build_supplementary_figures.py)
and its source hashes and evidence notes are recorded in
`results/cross_model_pilot_synthesis/data/supplementary_figure_set_provenance.json`.

For a paper figure, keep both a vector PDF for LaTeX and a PNG preview when the plotting pipeline supports both. Use the existing redraw protocol in `gallery/redraw/` for the publication palette and typography.
