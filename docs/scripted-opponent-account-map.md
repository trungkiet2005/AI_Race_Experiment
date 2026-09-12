# Scripted-opponent campaign: account label map

The supplementary material prints the collection rotation for the
scripted-opponent campaign with anonymised account labels, because the review is
double blind and a benchmark account handle names a person. This file is the
mapping, and it is deliberately kept out of the submission bundle: the bundle
carries the supplementary document and nothing else.

Labels were assigned in the order the accounts first appear when the printed
table is read left to right, top to bottom, so the labels carry no information
about the handles themselves.

| Label | Account |
|---|---|
| Account A | `kit567` |
| Account B | `hunhtrungkit` |
| Account C | `tnkiet` |
| Account D | `trungkiet` |
| Account E | `foundnotkiet` |

The unlabelled values stay in each cell's own `collection_receipt.json` under
`results/frontier/scripted_opponent_campaign/`, which is where the analyser and
`scripts/verify_manuscript_claims.py` read them. Restore the handles in the
camera-ready artifact, when the paper is no longer anonymous.
