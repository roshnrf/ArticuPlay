"""Turns the raw UXSSD pull into labeled training examples.

Each row of uxssd-pronunciation-scores.csv judges one phone (velar/rhotic) in one
word in one real recording. This script slices out that word's audio segment
(using the matching .lab word-boundary file) and assigns a binary label:
1 = correct (primary_score >= 4), 0 = needs work (primary_score < 4).

HTK .lab time units are 100ns (divide by 1e7 for seconds) — confirmed against
real UltraSuite label files, not assumed.

Output: data/examples.csv (segment_path, speaker, phone_class, phone, word, label)
        data/segments/*.wav (one per training example)
"""
import csv
import wave
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
CSV_PATH = DATA_DIR / "uxssd-pronunciation-scores.csv"
LAB_DIR = DATA_DIR / "labels" / "word-labels"
AUDIO_DIR = DATA_DIR / "core-uxssd" / "core"
SEGMENTS_DIR = DATA_DIR / "segments"

HTK_UNITS_PER_SECOND = 1e7


def utt_to_wav_path(utt: str) -> Path:
    speaker, session, fileid = utt.split("-")
    return AUDIO_DIR / speaker / session / f"{fileid}.wav"


def parse_lab_file(utt: str) -> list[tuple[float, float, str]]:
    """Returns [(start_sec, end_sec, WORD), ...] for one utterance."""
    lab_path = LAB_DIR / f"{utt}.lab"
    entries = []
    for line in lab_path.read_text().splitlines():
        start, end, word = line.split()
        entries.append((int(start) / HTK_UNITS_PER_SECOND, int(end) / HTK_UNITS_PER_SECOND, word))
    return entries


def slice_wav(src_path: Path, start_sec: float, end_sec: float, out_path: Path) -> None:
    with wave.open(str(src_path), "rb") as src:
        params = src.getparams()
        frame_rate = params.framerate
        start_frame = int(start_sec * frame_rate)
        end_frame = int(end_sec * frame_rate)
        src.setpos(start_frame)
        frames = src.readframes(end_frame - start_frame)

    with wave.open(str(out_path), "wb") as out:
        out.setparams(params)
        out.writeframes(frames)


def build() -> None:
    SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))

    output_rows = []
    skipped = 0
    for i, row in enumerate(rows):
        utt = row["utt"]
        target_word = row["word"]
        wav_path = utt_to_wav_path(utt)

        word_boundaries = parse_lab_file(utt)
        match = next((w for w in word_boundaries if w[2] == target_word), None)
        if match is None:
            skipped += 1
            continue
        start_sec, end_sec, _ = match

        segment_filename = f"{utt}_{target_word}_{i}.wav"
        segment_path = SEGMENTS_DIR / segment_filename
        slice_wav(wav_path, start_sec, end_sec, segment_path)

        speaker = utt.split("-")[0]
        label = 1 if float(row["primary_score"]) >= 4 else 0

        output_rows.append(
            {
                "segment_path": str(segment_path.relative_to(DATA_DIR)),
                "speaker": speaker,
                "utt": utt,
                "phone_class": row["phone_class"],
                "phone": row["phone"],
                "word": target_word,
                "duration_sec": round(end_sec - start_sec, 3),
                "primary_score": row["primary_score"],
                "label": label,
            }
        )

    out_csv = DATA_DIR / "examples.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"built {len(output_rows)} examples, skipped {skipped} (word not found in lab file)")
    print(f"written to {out_csv}")


if __name__ == "__main__":
    build()
