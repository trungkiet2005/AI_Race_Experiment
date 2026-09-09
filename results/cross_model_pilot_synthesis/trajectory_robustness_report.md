# Trajectory robustness follow-up

Evidence class: diagnostic. This report is a robustness audit of descriptive
trajectory summaries, not a new behavioural claim and not a causal test.

## Source and storage

The analysis reconstructs 760 first-five-round trajectories from the existing
raw human table and seven model baseline tables. The source hashes, package
versions, grid, and output names are in
`data/trajectory_clustering_robustness.json`. The sample-size-matched diversity
analysis is recorded in `data/trajectory_diversity_rarefaction.csv` with its
provenance receipt in `data/trajectory_diversity_rarefaction.json`.

## HDBSCAN sweep

The sweep covers `min_cluster_size` values 10, 15, 20, and 25, crossed with
`min_samples` values 3, 6, and 9. Across the 12 settings, the recovered number
of clusters ranges from 7 to 18, the pooled noise fraction ranges from 6.4% to
30.4%, and adjusted Rand agreement with the fixed reference setting
(`min_cluster_size=20`, `min_samples=6`) ranges from 0.589 to 1.000.

The result is therefore sensitive in its exact partition and noise assignment.
The broad observation that several model populations are concentrated into
small observed trajectory supports while the human sample occupies more of the
feature space is not tied to one setting, but it remains descriptive and
sample-scoped. The full population-specific values are in
`data/trajectory_hdbscan_robustness.csv`.

## t-SNE sweep

The visualization stress test uses perplexities 5, 15, 30, and 50 with three
random seeds per perplexity and random initialization. Trustworthiness ranges
from 0.956 to 0.966. Fifteen-neighbour overlap with the perplexity-30,
seed-20260908 reference ranges from 0.453 to 1.000. The embedding is therefore
useful for visual exploration but not a stable metric space. Conclusions about
population diversity are taken from the embedding-free rarefaction table, not
from apparent two-dimensional separation. Full values are in
`data/trajectory_tsne_robustness.csv`.

## Boundary

The sample-size-matched Hill q=1 results retain an explicit GPT-5.4 nano
exception: its effective paired-trajectory count is close to the rarefied human
reference at all three risk caps. The safe interpretation is model-specific
policy compression in the tested sample, not an unconditional claim that humans
are always more diverse than language models.
