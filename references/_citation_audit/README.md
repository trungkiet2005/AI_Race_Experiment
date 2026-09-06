# Citation audit source bundles

arXiv LaTeX source bundles retained while auditing the citations in `paper/`.
They are kept as sources rather than PDFs so a claim can be traced to the exact
sentence, table, or number a cited paper actually states, which a rendered PDF
makes harder to grep.

| arXiv ID | bundle | extracted |
|---|---|---|
| 2308.11483 | `2308.11483.tar.gz` | `2308.11483/` (ACL style) |
| 2310.11324 | `2310.11324.tar.gz` | `2310.11324/` (ICLR 2024 style) |
| 2311.10054 | `2311.10054.tar.gz` | `2311.10054/` (EMNLP 2024 style) |
| 2401.03729 | `2401.03729.tar.gz` | `2401.03729/` (ACL style) |

Each ID resolves at `https://arxiv.org/abs/<id>`. The venue column is read off the
style file shipped in the bundle, not off a published record, so treat it as the
template the authors submitted with rather than a confirmed acceptance.

This is reference material, not project code or data: nothing in `ai_race/`,
`analysis/`, or `results/` reads from this directory. The retained source study
itself lives in `references/papers/`, separately.

See `paper/CITATION_CHANGELOG.md` for what the audit changed.
