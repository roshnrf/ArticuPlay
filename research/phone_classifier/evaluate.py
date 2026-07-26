"""Aggregates LOSO-CV fold results into honest overall metrics: accuracy,
precision/recall, and a per-phone-class (velar vs rhotic) breakdown.

Note: the compare_ipa() baseline comparison from the plan needs one more step
not yet built — compare_ipa takes two IPA strings (target vs. what was said),
but this dataset gives audio + a clinician correctness score, not a text
transcript of the attempt. To compare fairly, each segment would need to be
transcribed first (Whisper's decoder, not just its encoder used here), then
IPA-compared against the target word. Flagging honestly rather than skipping
silently — this is the natural next step, not built in this pass.
"""
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_score, recall_score

DATA_DIR = Path(__file__).parent / "data"


def main() -> None:
    results = np.load(DATA_DIR / "loso_results.npy", allow_pickle=True)

    all_true = np.concatenate([r["y_true"] for r in results])
    all_pred = np.concatenate([r["y_pred"] for r in results])
    all_phone_class = np.concatenate([r["phone_class"] for r in results])

    overall_acc = (all_true == all_pred).mean()
    overall_precision = precision_score(all_true, all_pred, zero_division=0)
    overall_recall = recall_score(all_true, all_pred, zero_division=0)

    print("=== Overall (pooled across all 8 LOSO folds) ===")
    print(f"n = {len(all_true)}")
    print(f"accuracy  = {overall_acc:.3f}")
    print(f"precision = {overall_precision:.3f}  (of predicted-correct, how many really were)")
    print(f"recall    = {overall_recall:.3f}  (of truly-correct, how many were caught)")
    majority_baseline = max((all_true == 1).mean(), (all_true == 0).mean())
    print(f"majority-class baseline = {majority_baseline:.3f}  (what a trivial always-guess-majority model gets)")

    print("\n=== Per phone class ===")
    for phone_class in sorted(set(all_phone_class)):
        mask = all_phone_class == phone_class
        acc = (all_true[mask] == all_pred[mask]).mean()
        print(f"{phone_class:10s} n={mask.sum():4d}  accuracy={acc:.3f}")

    print("\n=== Per speaker (from training log) ===")
    for r in results:
        acc = (r["y_true"] == r["y_pred"]).mean()
        print(f"{r['held_out_speaker']:5s} n={r['n_test']:4d}  accuracy={acc:.3f}")


if __name__ == "__main__":
    main()
