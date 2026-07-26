"""Splits the 250 real product words into train (200) and held-out test (50)
sets — a genuine generalization test, unlike the previous rounds which tested
on the exact words trained on."""
import csv
import json
import random
import urllib.request
from pathlib import Path

BACKEND = "http://127.0.0.1:8000/api/v1"
DATA_DIR = Path(__file__).parent / "data"

random.seed(42)


def fetch_all_words():
    words = []
    for level in range(1, 6):
        req = urllib.request.Request(f"{BACKEND}/content/words/{level}?lang=en")
        with urllib.request.urlopen(req) as resp:
            for w in json.loads(resp.read()):
                w["level"] = level
                words.append(w)
    return words


def main():
    words = fetch_all_words()
    by_level = {}
    for w in words:
        by_level.setdefault(w["level"], []).append(w)

    held_out = []
    train_words = []
    for level, level_words in by_level.items():
        random.shuffle(level_words)
        held_out.extend(level_words[:10])  # 10 per level = 50 total
        train_words.extend(level_words[10:])

    with open(DATA_DIR / "heldout_words.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["word", "level"])
        writer.writeheader()
        writer.writerows([{"word": w["word"], "level": w["level"]} for w in held_out])

    with open(DATA_DIR / "trainvocab_words.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["word", "level"])
        writer.writeheader()
        writer.writerows([{"word": w["word"], "level": w["level"]} for w in train_words])

    print(f"held out: {len(held_out)} words (never used in training)")
    print(f"train vocab: {len(train_words)} words")


if __name__ == "__main__":
    main()
