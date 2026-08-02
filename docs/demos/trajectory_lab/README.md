# Trajectory Lab

Standalone, offline demo of paired trajectory divergence in the Qwen T=0
context-skin pilot. Open `index.html` directly or serve the repository root with
any static HTTP server.

The demo contains 36 high-information exemplars generated from immutable turn
logs. It does not call a model or external service. The visible claim boundary
is intentional: this is diagnostic paired evidence, not confirmatory causal
mediation.

Rebuild the embedded payload with:

```bash
python results/scripts/build_trajectory_demo.py
```
