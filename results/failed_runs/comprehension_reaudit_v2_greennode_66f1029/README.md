# Failed comprehension re-audit attempt

`allocator_batch8/` preserves the first block-1 attempt. The coordinator
failed closed when Qwen2.5-7B on the 20 GB MIG returned a PyTorch/NVML caching
allocator error at batch size 8. No behavioral or comprehension result from
this directory is valid evidence.

The successful retry used Qwen batch size 1 and is stored under
`results/open_source/comprehension_reaudit_v2_greennode_66f1029/block1/`.
