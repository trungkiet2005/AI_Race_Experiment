# Kaggle cross-model scaffold smoke v4 — failed run receipt

- Kernel: `daosyduyminh/ai-race-impact-admission`, version 4
- URL: https://www.kaggle.com/code/daosyduyminh/ai-race-impact-admission
- Observed terminal status: `KernelWorkerStatus.ERROR`
- Intended scope: Qwen2.5-7B, English, smoke profile, 160 comprehension requests
- Evidence status: **failed; zero admitted requests and zero gameplay claims**

Version 4 added support for Kaggle's flat directory-mode Dataset mount. The
kernel still terminated before publishing a result artifact. The output-log
endpoint returned HTTP 429 during the release audit, so the exact v4 traceback
is not claimed here. Versions 2 and 3 retain their downloaded traceback logs in
adjacent failed-run directories.

This receipt is deliberately separated from completed evidence. It must not be
used in effect estimates, cross-model coverage counts, or paper claims.
