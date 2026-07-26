"""Builds the UXTD manifest: (wav_path, transcript_text, speaker) for every
utterance, then a speaker-disjoint train/val/test split. 58 speakers is enough
for a standard split (unlike Phase 1's 8-speaker case, which needed full LOSO).
"""
import csv
import random
import re
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
CORE_DIR = DATA_DIR / "core-uxtd" / "core"

random.seed(42)

# roughly 78/10/12 split by speaker count
VAL_SPEAKERS = 6
TEST_SPEAKERS = 7

# Real finding (2026-07-25): a first regex filter (teaching|pre-test|post-test)
# looked sufficient on a small 8-word spot check, but testing at full scale
# (250 real product words) revealed it missed a second contamination category:
# standalone phonetic-articulation terminology ("linguolabial", "velar",
# "alveolar", "cardinal-1", "creaky-voice", "swallow", etc.) with no compound
# marker to regex-match on. Fixed properly this time: audited ALL 100 unique
# prompt values across the entire corpus (only 100 exist total) and built an
# exact-match denylist from manual review, instead of another regex guess.
_ADMIN_LABELS = {
    "labiodental-nasal", "linguolabial", "lateral", "creaky-voice",
    "spread-lips-post-alveolar-fricative", "cardinal-4", "alveolar",
    "cardinal-1", "velar", "swallow", "cough",
}
_ADMIN_LABEL_PATTERN = re.compile(r"\b(teaching|pre-test|post-test)\b", re.IGNORECASE)


def is_admin_label(text: str) -> bool:
    return text.strip().lower() in _ADMIN_LABELS or bool(_ADMIN_LABEL_PATTERN.search(text))


def build_manifest() -> list[dict]:
    rows = []
    skipped = 0
    for txt_path in sorted(CORE_DIR.rglob("*.txt")):
        wav_path = txt_path.with_suffix(".wav")
        if not wav_path.exists():
            continue
        text = txt_path.read_text().splitlines()[0].strip()
        if is_admin_label(text):
            skipped += 1
            continue
        speaker = wav_path.relative_to(CORE_DIR).parts[0]
        rows.append({"wav_path": str(wav_path.relative_to(DATA_DIR)), "text": text, "speaker": speaker})
    print(f"filtered out {skipped} session-administration labels (non-speech content)")
    return rows


def split_by_speaker(rows: list[dict]) -> dict[str, list[dict]]:
    speakers = sorted(set(r["speaker"] for r in rows))
    random.shuffle(speakers)

    test_speakers = set(speakers[:TEST_SPEAKERS])
    val_speakers = set(speakers[TEST_SPEAKERS:TEST_SPEAKERS + VAL_SPEAKERS])
    train_speakers = set(speakers[TEST_SPEAKERS + VAL_SPEAKERS:])

    assert not (test_speakers & val_speakers & train_speakers), "overlap detected"
    assert test_speakers.isdisjoint(val_speakers)
    assert test_speakers.isdisjoint(train_speakers)
    assert val_speakers.isdisjoint(train_speakers)

    return {
        "train": [r for r in rows if r["speaker"] in train_speakers],
        "val": [r for r in rows if r["speaker"] in val_speakers],
        "test": [r for r in rows if r["speaker"] in test_speakers],
    }


def main() -> None:
    rows = build_manifest()
    print(f"built manifest: {len(rows)} utterances across {len(set(r['speaker'] for r in rows))} speakers")

    splits = split_by_speaker(rows)
    for name, split_rows in splits.items():
        speakers = sorted(set(r["speaker"] for r in split_rows))
        out_path = DATA_DIR / f"{name}.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["wav_path", "text", "speaker"])
            writer.writeheader()
            writer.writerows(split_rows)
        print(f"{name}: {len(split_rows)} utterances, {len(speakers)} speakers -> {out_path}")


if __name__ == "__main__":
    main()
