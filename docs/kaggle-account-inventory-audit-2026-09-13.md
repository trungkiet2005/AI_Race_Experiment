# Kaggle account inventory audit, 2026-09-13

The inventory was checked without printing tokens. API-token readability was
separated from Benchmark auth: a token can list tasks while still failing the
Benchmark auth exchange needed to obtain `MODEL_PROXY_*` credentials.

## Standard runtime profiles

| Source | Identity | Benchmark use |
|---|---|---|
| `runtime_profiles/foundnotkiet` | `foundnotkiet` | auth passed; successful campaign receipts |
| `runtime_profiles/hunhtrungkit` | `hunhtrungkit` | auth passed; successful campaign receipts |
| `runtime_profiles/kit567` | `kit567` | auth passed; successful campaign receipts |
| `runtime_profiles/tnkiet` | `tnkiet` | auth passed; successful campaign receipts |
| `runtime_profiles/trungkiet` | `trungkiet` | auth passed; successful campaign receipts |

## Raw-token sources

| Source | Identity | Benchmark auth re-check | Campaign outcome |
|---|---|---|---|
| `kaggle-api-3/acc06.txt` | `minh2duy` | passed | successful receipts and one quota refusal |
| `kaggle-api-3/acc07.txt` | `boymagic` | passed | successful receipts |
| `kaggle-api-3/acc08.txt` | `trngthtnhi` | passed | successful receipts |
| `kaggle-api-3/acc09.txt` | `osduyminh` | passed | quota refusal; no successful receipt |
| `kaggle-api-3/acc10.txt` | `trnnguynchis` | failed with 403 | excluded; no model output |
| `kaggle-api-3/acc11.txt` | `trngbotrn` | passed | successful receipts |
| `kaggle-api-2/acc5.md` | `daosyduyminh` | passed | successful final `AS@0.6` receipt |
| `kaggle-api/chisboiz.txt` | `chisboiz` | passed | successful final `CAS@0.9` receipt |
| `kaggle-api/chunaiu.txt` | `chunaiu` | passed | successful final `CS@0.1` receipt |
| `kaggle-api/kakagotto.txt` | `kakagotto` | passed | not assigned; no need to spend quota |
| `kaggle-api/tonngohan.txt` | `tonngohan` | passed | not assigned; no need to spend quota |
| `kaggle-api/trunkdabest.txt` | `trunkdabest` | passed | not assigned; no need to spend quota |
| `kaggle-api/vinhdinhthien.txt` | `vinhdinhthien` | passed | not assigned; no need to spend quota |
| `kaggle-api/chiboiz.txt` | `chiboiz` | failed with 403 | excluded; no model output |
| `kaggle-api/chinguyentran.txt` | `chinguyentran` | failed with 403 | excluded; no model output |

`kaggle-api-2/acc1.md` through `acc4.md` and the duplicate root JSON files map
to identities already listed above; they are not counted as new people or
spent on duplicate cells. The root inventory therefore contains 20 unique
identities, 17 with a successful Benchmark auth exchange in this audit, while
12 distinct identities have at least one successful campaign receipt.

No credential value is stored in this repository. The task code cannot attest
the executing account, so account identity is recorded by the ingestion receipt
and fixed assignment document; it is not falsely represented as a server-side
manifest field.

