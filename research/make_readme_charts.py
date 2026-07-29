"""Generates the model-comparison and dev-progress charts for the README.
Every number here traces to a real logged result in tasks/lessons.md — no
invented figures. Re-run after any future retrain to keep the charts honest."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "/mnt/c/Users/rosha/Documents/sw_2/docs/assets"

plt.rcParams.update({
    "font.size": 12,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.edgecolor": "#d1d5db",
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
    "axes.titlepad": 14,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "ytick.left": False,
})

GRAY = "#cbd5e1"
BLUE = "#2563eb"
BAR_WIDTH = 0.5


def _label(ax, bar, value, unit="%"):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{value}{unit}",
            ha="center", va="bottom", fontsize=12, fontweight="bold", color="#1f2937")


def _style_bar_axes(ax, ymax):
    ax.set_ylim(0, ymax)
    ax.set_yticks([])
    ax.grid(axis="y", color="#f1f5f9", linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0, labelsize=11.5)


def chart_phone_classifier():
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    versions = ["v1 — frozen encoder", "v2 — LoRA unfrozen"]
    scores = [60.9, 67.9]
    bars = ax.bar(versions, scores, color=[GRAY, BLUE], width=BAR_WIDTH, zorder=3)
    for bar, score in zip(bars, scores):
        _label(ax, bar, score)
    _style_bar_axes(ax, 82)
    ax.set_title("Phoneme classifier — velar/rhotic correctness\nLeave-one-speaker-out CV, real UXSSD child speech")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/phone_classifier_progress.png", dpi=200)
    plt.close(fig)


def chart_in_sample():
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    labels = ["Base Whisper-small", "Fine-tuned (250 vocab)"]
    scores = [87.6, 94.8]
    bars = ax.bar(labels, scores, color=[GRAY, BLUE], width=BAR_WIDTH, zorder=3)
    for bar, score in zip(bars, scores):
        _label(ax, bar, score)
    _style_bar_axes(ax, 102)
    ax.set_title("In-sample accuracy — same vocab used in training\nreal compare_ipa phoneme scoring, n=250 words")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/in_sample_accuracy.png", dpi=200)
    plt.close(fig)


def chart_generalization():
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    labels = ["Base\nWhisper-small", "Fine-tuned\nseed 42", "Fine-tuned\nseed 43", "Fine-tuned\nseed 44"]
    scores = [80.0, 96.0, 98.2, 96.0]
    colors = [GRAY, BLUE, BLUE, BLUE]
    bars = ax.bar(labels, scores, color=colors, width=0.55, zorder=3)
    for bar, score in zip(bars, scores):
        _label(ax, bar, score)
    _style_bar_axes(ax, 108)
    ax.set_title("Held-out generalization — 30 words never seen in training\nnot vocabulary memorization")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/generalization_holdout.png", dpi=200)
    plt.close(fig)


def chart_error_type():
    fig, ax = plt.subplots(figsize=(7, 4.4))
    categories = ["False-accept\n(missed a real error)", "False-reject\n(flagged a correct one)"]
    stage1 = [28, 32]   # target-word bias (old)
    stage2 = [15, 44]   # unbiased decoding
    stage3 = [3, 46]    # + phone classifier (current)
    x = [0, 1.3]
    width = 0.3
    bars1 = ax.bar([i - width for i in x], stage1, width, label="Target-word bias (old)", color="#fb923c", zorder=3)
    bars2 = ax.bar(x, stage2, width, label="Unbiased decoding", color="#94a3b8", zorder=3)
    bars3 = ax.bar([i + width for i in x], stage3, width, label="+ Phone classifier (current)", color=BLUE, zorder=3)
    for bars, vals in [(bars1, stage1), (bars2, stage2), (bars3, stage3)]:
        for bar, v in zip(bars, vals):
            _label(ax, bar, v)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    _style_bar_axes(ax, 58)
    ax.set_title("Error-type rates on real disordered child speech\nUXSSD, clinician-labeled, n=200")
    ax.legend(fontsize=9.5, loc="upper center", frameon=False)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/error_type_rates.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    chart_phone_classifier()
    chart_in_sample()
    chart_generalization()
    chart_error_type()
    print(f"charts saved to {OUT_DIR}")
