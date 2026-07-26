"""Generates synthetic audio for ALL 250 real StoryWeaver product words
(levels 1-5, not just the 100 phrase/sentence ones from before) — directly
closes the vocabulary-mismatch gap: UXTD's word lists are different words
than our product's, so the model never saw most of our actual vocabulary.
Same honest limitation as before: synthetic TTS voice, not real child speech,
teaches vocabulary/structure exposure, not real acoustic variation.
"""
import csv
import json
import subprocess
import urllib.request
from pathlib import Path

BACKEND = "http://127.0.0.1:8000/api/v1"
DATA_DIR = Path(__file__).parent / "data"
AUGMENT_DIR = DATA_DIR / "augment_full_vocab"


def fetch_all_words() -> list[dict]:
    words = []
    for level in range(1, 6):
        req = urllib.request.Request(f"{BACKEND}/content/words/{level}?lang=en")
        with urllib.request.urlopen(req) as resp:
            words.extend(json.loads(resp.read()))
    return words


def synthesize(text: str, out_path: Path) -> None:
    body = json.dumps({"word": text, "language": "en"}).encode()
    req = urllib.request.Request(
        f"{BACKEND}/session/tts", data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        out_path.write_bytes(resp.read())


def main() -> None:
    AUGMENT_DIR.mkdir(parents=True, exist_ok=True)
    words = fetch_all_words()
    print(f"generating synthetic audio for all {len(words)} real product words...")

    manifest_rows = []
    for i, entry in enumerate(words):
        mp3_path = AUGMENT_DIR / f"aug_{i}.mp3"
        wav_path = AUGMENT_DIR / f"aug_{i}.wav"
        synthesize(entry["word"], mp3_path)
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(mp3_path), "-ar", "16000", "-ac", "1", str(wav_path), "-loglevel", "error"],
            check=True,
        )
        manifest_rows.append({"wav_path": str(wav_path.relative_to(DATA_DIR)), "text": entry["word"], "speaker": "SYNTH"})
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(words)}")

    with open(DATA_DIR / "augment_full_vocab.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["wav_path", "text", "speaker"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"done — {len(manifest_rows)} synthetic examples saved to data/augment_full_vocab.csv")


if __name__ == "__main__":
    main()
