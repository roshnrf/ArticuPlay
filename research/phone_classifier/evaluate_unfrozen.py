"""Same evaluation methodology as evaluate.py (Phase 1's frozen baseline),
applied to the LoRA-adapted encoder results — apples-to-apples comparison."""
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_score, recall_score

DATA_DIR = Path(__file__).parent / "data"


def main() -> None:
    frozen = np.load(DATA_DIR / "loso_results.npy", allow_pickle=True)
    unfrozen = np.load(DATA_DIR / "loso_results_unfrozen.npy", allow_pickle=True)

    for label, results in [("Phase 1 — frozen embeddings", frozen), ("Phase 2b — LoRA-adapted encoder", unfrozen)]:
        all_true = np.concatenate([r["y_true"] for r in results])
        all_pred = np.concatenate([r["y_pred"] for r in results])
        all_phone_class = np.concatenate([r["phone_class"] for r in results])

        acc = (all_true == all_pred).mean()
        precision = precision_score(all_true, all_pred, zero_division=0)
        recall = recall_score(all_true, all_pred, zero_division=0)

        print(f"=== {label} ===")
        print(f"n={len(all_true)}  accuracy={acc:.3f}  precision={precision:.3f}  recall={recall:.3f}")
        for phone_class in sorted(set(all_phone_class)):
            mask = all_phone_class == phone_class
            class_acc = (all_true[mask] == all_pred[mask]).mean()
            print(f"  {phone_class:10s} n={mask.sum():4d}  accuracy={class_acc:.3f}")
        print()

    frozen_acc = (np.concatenate([r["y_true"] for r in frozen]) == np.concatenate([r["y_pred"] for r in frozen])).mean()
    unfrozen_acc = (np.concatenate([r["y_true"] for r in unfrozen]) == np.concatenate([r["y_pred"] for r in unfrozen])).mean()
    print(f"Improvement from unfreezing (LoRA): {unfrozen_acc - frozen_acc:+.3f} ({(unfrozen_acc - frozen_acc) * 100:+.1f} points)")


if __name__ == "__main__":
    main()
