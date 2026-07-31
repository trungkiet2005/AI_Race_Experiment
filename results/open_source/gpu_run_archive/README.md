# GreenNode GPU result archive

This directory is the immutable public handoff of every AI Race result bundle
found on the two-pod shared volume on 2026-07-31. Both pods exposed the same
files, so each shared-disk artifact is stored once rather than duplicated by
pod.

The pod inventory contained 371 prompt/persona/surface output files and 59
game-understanding run files. The archives account for all 371 prompt-side
files and all 50 scientific game-understanding files; the other nine game files
were operational lane logs and are intentionally excluded.

The archive contains 11 gzip-compressed tar bundles (about 1.7 MB total):

| Study | Evidence class | Bundles |
|---|---|---|
| Persona sensitivity | three smoke raw bundles, pilot raw, pilot analysis | `smoke*.tar.gz`, `pilot-identified-t1-0-results.tar.gz`, `analysis-pilot-identified-t1-0.tar.gz` |
| Surface sensitivity | smoke raw, superseded v1 analysis, corrected v2 analysis, and pilot raw/analysis | `surface-smoke-*.tar.gz`, `surface-pilot-*.tar.gz` |
| Game understanding | corrected smoke, rejected earlier smoke, and admitted pilot raw outputs | `game-understanding-results.tar.gz` |

The prompt-sensitivity source revision
`c6a3e541b3eee90c1ec53fb74469b2264405b171` had no result payloads, so it is
recorded as a zero-output attempt rather than represented by an empty archive.
Source snapshots and source archives are also excluded because the exact Git
revisions are already preserved in repository history.

Expanded, review-friendly derived tables are available beside this directory:

- `../prompt_sensitivity_pilot/`
- `../surface_sensitivity_pilot/`
- `../surface_sensitivity_smoke/`
- `../game_understanding_pilot/`

All are diagnostic smoke or pilot evidence. None is confirmatory evidence.

## Integrity check

From a clean clone, run:

```bash
python results/scripts/audit_gpu_archives.py \
  --archive-dir results/open_source/gpu_run_archive
```

The command fails if an archive is missing or added, if any SHA-256 differs,
if a tar member has an unsafe path, if JSON metadata is malformed, or if the
archive contains duplicate member names. The expected machine-readable result
is committed as `archive_ledger.json`.

## Public boundary

These bundles retain scientific prompts, model responses, outcome tables, and
run manifests needed for reproduction. Pod access logs, PID files, credentials,
IP addresses, source snapshots, model caches, and third-party paper copies are
not included.
