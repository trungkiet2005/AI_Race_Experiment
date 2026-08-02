# Comprehension re-audit v2 — GreenNode

This directory is the canonical repository copy of the completed atomic
comprehension re-audit executed from source commit `66f1029`.

- `block1/`: Qwen2.5-7B on the 20 GB MIG and Mistral-7B on the 40 GB MIG.
- `block2/`: lane swap, with Mistral-7B on 20 GB and Qwen2.5-7B on 40 GB.
- Each block contains the complete raw response log, mailbox audit, summary,
  and provenance-rich run manifest.
- The initial Qwen batch-8 allocator failure is retained separately under
  `results/failed_runs/comprehension_reaudit_v2_greennode_66f1029/`.

Both completed blocks passed coverage and rescore integrity for 274/274
responses. They remain `diagnostic_unadmitted`: the v2 rules context still
displayed a canonical round-one state before some independent hypothetical
questions. The context-clean v3 protocol supersedes v2 for future admission.

Calculator rows measure uptake of disclosed verified arithmetic, not unaided
understanding. Comparisons must be item-balanced because the unaided and
calculator arms have different numbers of surface realizations per item.
