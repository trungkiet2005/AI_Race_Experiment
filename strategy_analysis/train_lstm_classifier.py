"""Optional LSTM strategy classifier, trained on synthetic AS/AU/CS/CAS data.

This mirrors the training pattern used for the sibling IPD project's
30-round strategy classifiers (D:/AI_RESEARCH/ClusteringResearch/scripts/
train_lstm_30round_v2.py: Masking -> LSTM(32) -> Dropout -> LSTM(16) ->
Dropout -> softmax, trained with class-balanced oversampling and early
stopping), adapted to this project's variable-horizon Safe/Unsafe races.

It is **not** a replacement for ``strategy_analysis.classify`` (the nearest
Hamming-distance rule matches the paper's four canonical strategies exactly
and is the baseline used for confirmatory descriptive statistics). This
script exists to measure whether a learned classifier tolerates per-round
execution noise better than exact rule matching — useful when the eventual
target is real, imperfectly-rule-following LLM trajectories rather than the
noiseless synthetic dataset. Its predictions must not be substituted into
analysis pipelines that require the paper-faithful nearest-strategy rule.

Requires the optional ``strategy-ml`` extra (``pip install -e ".[strategy-ml]"``):
tensorflow, scikit-learn, imbalanced-learn. The dataset is small (thousands of
short sequences) and training runs in well under a minute on CPU; no GPU or
Kaggle execution is required for this script.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

try:
    from imblearn.over_sampling import RandomOverSampler
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise SystemExit(
        "train_lstm_classifier.py requires the optional 'strategy-ml' extra: "
        'pip install -e ".[strategy-ml]"'
    ) from exc

PAD_VALUE = -1.0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def build_features(
    records: list[dict[str, Any]], max_len: int
) -> tuple[np.ndarray, np.ndarray]:
    """Encode each trajectory as a (max_len, 2) array of (own, opponent) actions.

    Timesteps beyond a race's realised horizon are filled with ``PAD_VALUE`` in
    both channels and masked out by the model's ``Masking`` layer.
    """

    features = np.full((len(records), max_len, 2), PAD_VALUE, dtype=float)
    labels = np.empty(len(records), dtype=object)
    for i, record in enumerate(records):
        own = record["own_actions"][:max_len]
        opp = record["opponent_actions"][:max_len]
        for t, (a, b) in enumerate(zip(own, opp)):
            features[i, t, 0] = float(a)
            features[i, t, 1] = float(b)
        labels[i] = record["true_strategy"]
    return features, labels


def build_model(n_classes: int, max_len: int) -> "Sequential":
    model = Sequential(
        [
            Masking(mask_value=PAD_VALUE, input_shape=(max_len, 2)),
            LSTM(32, activation="relu", return_sequences=True),
            Dropout(0.2),
            LSTM(16, activation="relu"),
            Dropout(0.2),
            Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=5e-4),
        metrics=["accuracy"],
    )
    return model


def train(
    dataset_path: Path,
    *,
    max_len: int,
    epochs: int,
    batch_size: int,
    model_out: Path,
    encoder_out: Path,
) -> None:
    records = _read_jsonl(dataset_path)
    if not records:
        raise ValueError(f"{dataset_path}: no records found")

    X, y_str = build_features(records, max_len)
    print(f"X shape: {X.shape}")

    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y_str)
    n_classes = len(label_encoder.classes_)
    print(f"classes ({n_classes}): {list(label_encoder.classes_)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.25, random_state=1, stratify=y_enc
    )
    y_test_cat = to_categorical(y_test, num_classes=n_classes)
    print(f"train: {len(X_train)} | test: {len(X_test)}")

    X_train_flat = X_train.reshape(len(X_train), -1)
    ros = RandomOverSampler(random_state=42)
    X_ros_flat, y_ros = ros.fit_resample(X_train_flat, y_train)
    X_ros = X_ros_flat.reshape(len(X_ros_flat), max_len, 2)
    y_ros_cat = to_categorical(y_ros, num_classes=n_classes)
    print(f"oversampled train: {len(X_ros)}")

    model = build_model(n_classes, max_len)
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=4, restore_best_weights=True, verbose=1
    )
    model.fit(
        X_ros,
        y_ros_cat,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test_cat),
        callbacks=[early_stopping],
        verbose=2,
    )
    train_loss, train_acc = model.evaluate(X_ros, y_ros_cat, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"train acc: {train_acc * 100:.2f}% | test acc: {test_acc * 100:.2f}%")

    model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_out)
    encoder_out.parent.mkdir(parents=True, exist_ok=True)
    with encoder_out.open("wb") as handle:
        pickle.dump(label_encoder, handle)
    print(f"saved model   -> {model_out}")
    print(f"saved encoder -> {encoder_out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="JSONL from generate_dataset.py")
    parser.add_argument("--max-len", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("strategy_analysis/artifacts/lstm_ascsaucas.keras"),
    )
    parser.add_argument(
        "--encoder-out",
        type=Path,
        default=Path("strategy_analysis/artifacts/label_encoder_ascsaucas.pkl"),
    )
    args = parser.parse_args(argv)

    train(
        args.dataset,
        max_len=args.max_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_out=args.model_out,
        encoder_out=args.encoder_out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
