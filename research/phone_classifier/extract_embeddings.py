"""Extracts a fixed-size embedding per audio segment using Whisper-small's
frozen encoder (no fine-tuning — 864 examples is too little to safely fine-tune
an encoder end-to-end, this is the standard linear-probe-on-frozen-features
pattern instead). Mean-pools over the time dimension to get one vector per clip.

Output: data/embeddings.npy (N x hidden_dim), data/labels.npy (N,), data/speakers.npy (N,)
"""
import csv
import wave

import numpy as np
import torch
import torchaudio
from transformers import WhisperFeatureExtractor, WhisperModel

DATA_DIR = __import__("pathlib").Path(__file__).parent / "data"
EXAMPLES_CSV = DATA_DIR / "examples.csv"
WHISPER_SAMPLE_RATE = 16000

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_audio_resampled(path: str) -> torch.Tensor:
    """Reads a wav file directly (torchaudio.load() now requires the separate
    torchcodec package we don't otherwise need) and resamples to Whisper's rate."""
    with wave.open(path, "rb") as f:
        sr = f.getframerate()
        n_channels = f.getnchannels()
        raw = f.readframes(f.getnframes())

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    waveform = torch.from_numpy(samples).unsqueeze(0)
    if sr != WHISPER_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, WHISPER_SAMPLE_RATE)
    return waveform.squeeze(0)


def main() -> None:
    with open(EXAMPLES_CSV) as f:
        rows = list(csv.DictReader(f))

    print(f"loading Whisper-small encoder on {device}...")
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    model = WhisperModel.from_pretrained("openai/whisper-small").encoder.to(device)
    model.eval()

    embeddings = []
    labels = []
    speakers = []
    phone_classes = []

    with torch.no_grad():
        for i, row in enumerate(rows):
            audio_path = DATA_DIR / row["segment_path"]
            waveform = load_audio_resampled(str(audio_path))

            inputs = feature_extractor(
                waveform.numpy(), sampling_rate=WHISPER_SAMPLE_RATE, return_tensors="pt"
            )
            input_features = inputs.input_features.to(device)

            encoder_output = model(input_features).last_hidden_state  # (1, T, hidden_dim)
            pooled = encoder_output.mean(dim=1).squeeze(0).cpu().numpy()  # (hidden_dim,)

            embeddings.append(pooled)
            labels.append(int(row["label"]))
            speakers.append(row["speaker"])
            phone_classes.append(row["phone_class"])

            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(rows)} embedded")

    np.save(DATA_DIR / "embeddings.npy", np.stack(embeddings))
    np.save(DATA_DIR / "labels.npy", np.array(labels))
    np.save(DATA_DIR / "speakers.npy", np.array(speakers))
    np.save(DATA_DIR / "phone_classes.npy", np.array(phone_classes))
    print(f"done — {len(embeddings)} embeddings, dim={embeddings[0].shape[0]}")


if __name__ == "__main__":
    main()
