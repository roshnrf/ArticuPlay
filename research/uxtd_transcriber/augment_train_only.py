"""Generates synthetic audio for only the 200 train-split words (holds out
50 words entirely for a genuine generalization test)."""
import csv
import json
import subprocess
import urllib.request
from pathlib import Path

BACKEND = "http://127.0.0.1:8000/api/v1"
DATA_DIR = Path(__file__).parent / "data"
AUGMENT_DIR = DATA_DIR / "augment_trainvocab"


def synthesize(text: str, out_path: Path) -> None:
    body = json.dumps({"word": text, "language": "en"}).encode()
    req = urllib.request.Request(
        f"{BACKEND}/session/tts", data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        out_path.write_bytes(resp.read())


def main() -> None:
    AUGMENT_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_DIR / "trainvocab_words.csv") as f:
        words = list(csv.DictReader(f))
    print(f"generating synthetic audio for {len(words)} train-split words...")

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

    with open(DATA_DIR / "augment_trainvocab.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["wav_path", "text", "speaker"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"done — {len(manifest_rows)} examples saved")


if __name__ == "__main__":
    main()
