"""Trains a phone-correctness classifier (logistic regression on frozen Whisper
embeddings) using Leave-One-Speaker-Out cross-validation.

Only 8 speakers exist. A random split risks the same voice appearing in both
train and test, which would let the model partly memorize speaker identity
rather than learn genuine phone-correctness signal. LOSO-CV (train on 7,
test on the 1 held out, repeat for each speaker) is the standard approach
for datasets this small — not optional here.
"""
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).parent / "data"


def run_loso_cv() -> list[dict]:
    embeddings = np.load(DATA_DIR / "embeddings.npy")
    labels = np.load(DATA_DIR / "labels.npy")
    speakers = np.load(DATA_DIR / "speakers.npy")
    phone_classes = np.load(DATA_DIR / "phone_classes.npy")

    unique_speakers = sorted(set(speakers))
    fold_results = []

    for held_out in unique_speakers:
        train_mask = speakers != held_out
        test_mask = speakers == held_out

        assert not (train_mask & test_mask).any(), "train/test overlap — should be impossible"
        assert set(speakers[train_mask]).isdisjoint(set(speakers[test_mask])), "speaker leaked across fold"

        scaler = StandardScaler()
        X_train = scaler.fit_transform(embeddings[train_mask])
        X_test = scaler.transform(embeddings[test_mask])
        y_train, y_test = labels[train_mask], labels[test_mask]

        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        fold_results.append(
            {
                "held_out_speaker": held_out,
                "y_true": y_test,
                "y_pred": y_pred,
                "phone_class": phone_classes[test_mask],
                "n_test": len(y_test),
            }
        )
        acc = (y_pred == y_test).mean()
        print(f"speaker {held_out} held out: n={len(y_test)}, accuracy={acc:.3f}")

    return fold_results


if __name__ == "__main__":
    results = run_loso_cv()
    np.save(DATA_DIR / "loso_results.npy", np.array(results, dtype=object))
    print(f"\nLOSO-CV complete across {len(results)} folds, saved to data/loso_results.npy")
