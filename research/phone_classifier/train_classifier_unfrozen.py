"""Phase 2b: does letting Whisper-small's encoder actually adapt (via LoRA, not
full unfreezing — 864 examples across 8 speakers is still too little to safely
unfreeze all ~90M encoder params) beat Phase 1's frozen-embedding baseline (60.9%)?

Same 864 real examples, same Leave-One-Speaker-Out CV structure as Phase 1,
so the comparison is apples-to-apples — same speakers, same folds, only the
encoder-adaptation strategy differs.
"""
import csv
import wave
from pathlib import Path

import numpy as np
import torch
import torch.distributed.tensor  # noqa: F401 — must import before peft: peft checks this
# submodule's presence without importing it itself, and torch 2.13 doesn't
# auto-expose it as a torch.distributed attribute unless explicitly imported first.
import torch.nn as nn
import torchaudio
from peft import LoraConfig, get_peft_model
from transformers import WhisperFeatureExtractor, WhisperModel

DATA_DIR = Path(__file__).parent / "data"
EXAMPLES_CSV = DATA_DIR / "examples.csv"
WHISPER_SAMPLE_RATE = 16000
EPOCHS = 3
LR = 1e-4

device = "cuda" if torch.cuda.is_available() else "cpu"


class PhoneClassifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.config.d_model, 1)

    def forward(self, input_features):
        hidden = self.encoder(input_features).last_hidden_state  # (B, T, D)
        pooled = hidden.mean(dim=1)  # (B, D)
        return self.head(pooled).squeeze(-1)  # (B,)


def load_audio_resampled(path: str) -> torch.Tensor:
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


def make_lora_encoder():
    base = WhisperModel.from_pretrained("openai/whisper-small").encoder
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],  # standard LoRA targets for transformer attention
        lora_dropout=0.05,
    )
    return get_peft_model(base, lora_config)


def run_loso_cv():
    with open(EXAMPLES_CSV) as f:
        rows = list(csv.DictReader(f))

    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    unique_speakers = sorted(set(r["speaker"] for r in rows))
    fold_results = []

    for held_out in unique_speakers:
        train_rows = [r for r in rows if r["speaker"] != held_out]
        test_rows = [r for r in rows if r["speaker"] == held_out]
        assert not set(r["speaker"] for r in train_rows) & {held_out}, "speaker leaked into train"

        model = PhoneClassifier(make_lora_encoder()).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        loss_fn = nn.BCEWithLogitsLoss()

        model.train()
        for epoch in range(EPOCHS):
            epoch_loss = 0.0
            for row in train_rows:
                waveform = load_audio_resampled(str(DATA_DIR / row["segment_path"]))
                inputs = feature_extractor(waveform.numpy(), sampling_rate=WHISPER_SAMPLE_RATE, return_tensors="pt")
                input_features = inputs.input_features.to(device)
                label = torch.tensor([float(row["label"])], device=device)

                optimizer.zero_grad()
                logit = model(input_features)
                loss = loss_fn(logit, label)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"  speaker {held_out} held out, epoch {epoch + 1}/{EPOCHS}: avg loss = {epoch_loss / len(train_rows):.4f}")

        model.eval()
        y_true, y_pred, phone_classes = [], [], []
        with torch.no_grad():
            for row in test_rows:
                waveform = load_audio_resampled(str(DATA_DIR / row["segment_path"]))
                inputs = feature_extractor(waveform.numpy(), sampling_rate=WHISPER_SAMPLE_RATE, return_tensors="pt")
                input_features = inputs.input_features.to(device)
                logit = model(input_features)
                pred = int(torch.sigmoid(logit).item() >= 0.5)
                y_true.append(int(row["label"]))
                y_pred.append(pred)
                phone_classes.append(row["phone_class"])

        y_true, y_pred = np.array(y_true), np.array(y_pred)
        acc = (y_true == y_pred).mean()
        print(f"speaker {held_out} held out: n={len(y_true)}, accuracy={acc:.3f}")

        fold_results.append(
            {"held_out_speaker": held_out, "y_true": y_true, "y_pred": y_pred, "phone_class": np.array(phone_classes), "n_test": len(y_true)}
        )

    return fold_results


if __name__ == "__main__":
    results = run_loso_cv()
    np.save(DATA_DIR / "loso_results_unfrozen.npy", np.array(results, dtype=object))
    print(f"\nLOSO-CV (LoRA-adapted encoder) complete across {len(results)} folds")
