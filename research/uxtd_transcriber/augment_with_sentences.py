"""Generates synthetic sentence-length training examples from StoryWeaver's
own Level 4/5 content (real product phrases/sentences), via the real
/session/tts endpoint, to directly address the diagnosed gap: the clean UXTD
dataset has almost no grammatical-sentence-structure content, only word lists
and nonsense syllables, which is why the fine-tuned model collapses on
Level 5 (full sentences) despite having no problem with raw audio duration.

Honest limitation, not hidden: this is synthetic TTS voice, not real child
speech. It teaches the decoder to handle longer grammatical text structurally
(the diagnosed missing exposure) — it does not teach real child sentence
acoustics, since no accessible real dataset has that (MyST corpus URL from
the original spec doc is unreachable, checked directly, not assumed).
"""
import json
import urllib.request
from pathlib import Path

BACKEND = "http://127.0.0.1:8000/api/v1"
DATA_DIR = Path(__file__).parent / "data"
AUGMENT_DIR = DATA_DIR / "augment_sentences"


def fetch_level_words(level: int) -> list[dict]:
    req = urllib.request.Request(f"{BACKEND}/content/words/{level}?lang=en")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def synthesize(word: str, out_path: Path) -> None:
    body = json.dumps({"word": word, "language": "en"}).encode()
    req = urllib.request.Request(
        f"{BACKEND}/session/tts", data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        out_path.write_bytes(resp.read())


def main() -> None:
    AUGMENT_DIR.mkdir(parents=True, exist_ok=True)
    entries = fetch_level_words(4) + fetch_level_words(5)
    print(f"generating synthetic audio for {len(entries)} Level 4/5 phrases/sentences...")

    rows = []
    for i, entry in enumerate(entries):
        mp3_path = AUGMENT_DIR / f"aug_{i}.mp3"
        synthesize(entry["word"], mp3_path)
        rows.append({"mp3_path": mp3_path, "text": entry["word"]})
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(entries)}")

    # convert to wav (Whisper pipeline expects wav via the wave module)
    import subprocess

    manifest_rows = []
    for i, row in enumerate(rows):
        wav_path = AUGMENT_DIR / f"aug_{i}.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(row["mp3_path"]), "-ar", "16000", "-ac", "1", str(wav_path), "-loglevel", "error"],
            check=True,
        )
        manifest_rows.append(
            {"wav_path": str(wav_path.relative_to(DATA_DIR)), "text": row["text"], "speaker": "SYNTH"}
        )

    import csv

    with open(DATA_DIR / "augment_sentences.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["wav_path", "text", "speaker"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"done — {len(manifest_rows)} synthetic sentence examples saved to data/augment_sentences.csv")


if __name__ == "__main__":
    main()
