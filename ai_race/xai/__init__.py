"""Mechanistic-XAI utilities for the AI Race experiments.

The activation-level pipeline in this package is deliberately separate from the
text/metadata surrogate models in ``results/scripts/explain_action_*.py``.
"""

from .activation_sae import (
    ACTIVATION_SAE_SCHEMA_VERSION,
    DecisionExample,
    build_capture_prefix,
    grouped_train_eval_split,
    load_decision_examples,
    race_prefix_components,
)

__all__ = [
    "ACTIVATION_SAE_SCHEMA_VERSION",
    "DecisionExample",
    "build_capture_prefix",
    "grouped_train_eval_split",
    "load_decision_examples",
    "race_prefix_components",
]
