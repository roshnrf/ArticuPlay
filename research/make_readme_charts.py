"""Generates the model-comparison and dev-progress charts for the README.
Every number here traces to a real logged result in tasks/lessons.md — no
invented figures. Re-run after any future retrain to keep the charts honest."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "/mnt/c/Users/rosha/Documents/sw_2/docs/assets"

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def chart_phone_classifier():
    fig, ax = plt.subplots(figsize=(5, 3.5))
    versions = ["v1\n(frozen encoder)", "v2\n(LoRA unfrozen)"]
    scores = [60.9, 67.9]
    bars = ax.bar(versions, scores, color=["#94a3b8", "#2563eb"], width=0.5)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, score + 1.5, f"{score}%", ha="center", fontweight="bold")
    ax.set_ylim(0, 85)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Phoneme classifier (velar/rhotic correctness)\nLeave-one-speaker-out CV, real UXSSD child speech")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/phone_classifier_progress.png", dpi=150)
    plt.close(fig)


def chart_in_sample():
    fig, ax = plt.subplots(figsize=(5, 3.5))
    labels = ["Base\nWhisper-small", "Fine-tuned\n(LoRA, 250 vocab)"]
    scores = [87.6, 94.8]
    bars = ax.bar(labels, scores, color=["#94a3b8", "#2563eb"], width=0.5)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, score + 1.5, f"{score}%", ha="center", fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Phoneme accuracy (%)")
    ax.set_title("In-sample accuracy (same vocab used in training)\nreal compare_ipa phoneme scoring, n=250 words")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/in_sample_accuracy.png", dpi=150)
    plt.close(fig)


def chart_generalization():
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    labels = ["Base\nWhisper-small", "Fine-tuned\n(seed 42)", "Fine-tuned\n(seed 43)", "Fine-tuned\n(seed 44)"]
    scores = [80.0, 96.0, 98.2, 96.0]
    colors = ["#94a3b8", "#2563eb", "#2563eb", "#2563eb"]
    bars = ax.bar(labels, scores, color=colors, width=0.55)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, score + 1.3, f"{score}%", ha="center", fontweight="bold")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Phoneme accuracy (%)")
    ax.set_title("Held-out generalization — 30 words never seen in training\n(the real test: not vocabulary memorization)")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/generalization_holdout.png", dpi=150)
    plt.close(fig)


def chart_error_type():
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    categories = ["False-accept\n(missed a real error)", "False-reject\n(flagged a correct one)"]
    with_bias = [28, 32]
    without_bias = [15, 44]
    x = range(len(categories))
    width = 0.32
    ax.bar([i - width / 2 for i in x], with_bias, width, label="With target-word bias (old)", color="#f97316")
    ax.bar([i + width / 2 for i in x], without_bias, width, label="Without bias (current)", color="#2563eb")
    for i, (a, b) in enumerate(zip(with_bias, without_bias)):
        ax.text(i - width / 2, a + 1, f"{a}%", ha="center", fontsize=9)
        ax.text(i + width / 2, b + 1, f"{b}%", ha="center", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(categories)
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 55)
    ax.set_title("Error-type rates on real disordered child speech\n(UXSSD, clinician-labeled, n=200) — honest, not the flattering number")
    ax.legend(fontsize=9, loc="upper center")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/error_type_rates.png", dpi=150)
    plt.close(fig)


def chart_dev_progress():
    fig, ax = plt.subplots(figsize=(7, 4.2))
    items = [
        ("Backend API (auth, children, sessions, scoring)", 100, "#22c55e"),
        ("Frontend UI (full drill loop, verified in-browser)", 100, "#22c55e"),
        ("ASR model: trained + integrated into production", 100, "#22c55e"),
        ("Phoneme-level scoring engine (compare_ipa)", 100, "#22c55e"),
        ("Real-world rigor checks (latency, variance, webm, error-type)", 100, "#22c55e"),
        ("Error-detection fix (route via phone classifier)", 20, "#f97316"),
        ("Multi-language support (English only today)", 5, "#ef4444"),
        ("Public deployment (live URL)", 0, "#ef4444"),
    ]
    labels = [i[0] for i in items][::-1]
    values = [i[1] for i in items][::-1]
    colors = [i[2] for i in items][::-1]
    bars = ax.barh(labels, values, color=colors, height=0.6)
    for bar, v in zip(bars, values):
        ax.text(min(v + 3, 96), bar.get_y() + bar.get_height() / 2, f"{v}%", va="center", fontsize=9, fontweight="bold")
    ax.set_xlim(0, 105)
    ax.set_xlabel("Complete (%)")
    ax.set_title("Development progress")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/dev_progress.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    chart_phone_classifier()
    chart_in_sample()
    chart_generalization()
    chart_error_type()
    chart_dev_progress()
    print(f"charts saved to {OUT_DIR}")
