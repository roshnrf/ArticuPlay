"""Trains the ONE deployable phone-classifier checkpoint, on all 864 real
examples (no LOSO holdout — that was for validating the approach, this is
for shipping it). Same architecture as train_classifier_unfrozen.py (the
67.9% LOSO-CV result), so that number is the honest expectation for this
model's real-world accuracy on new speakers.
"""
import csv
from pathlib import Path

import numpy as np
import torch
import torch.distributed.tensor  # noqa: F401 — must import before peft
import torch.nn as nn
import torchaudio
from peft import LoraConfig, get_peft_model
from transformers import WhisperFeatureExtractor, WhisperModel

from train_classifier_unfrozen import PhoneClassifier, load_audio_resampled, make_lora_encoder

DATA_DIR = Path(__file__).parent / "data"
EXAMPLES_CSV = DATA_DIR / "examples.csv"
OUTPUT_DIR = Path(__file__).parent / "phone_classifier_final"
WHISPER_SAMPLE_RATE = 16000
EPOCHS = 3
LR = 1e-4

device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    with open(EXAMPLES_CSV) as f:
        rows = list(csv.DictReader(f))
    print(f"training on all {len(rows)} real UXSSD examples (no holdout — this is the deploy model)")

    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    model = PhoneClassifier(make_lora_encoder()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for row in rows:
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
        print(f"  epoch {epoch + 1}/{EPOCHS}: avg loss = {epoch_loss / len(rows):.4f}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    model.encoder.save_pretrained(str(OUTPUT_DIR / "encoder_lora"))
    torch.save(model.head.state_dict(), OUTPUT_DIR / "head.pt")
    feature_extractor.save_pretrained(str(OUTPUT_DIR / "feature_extractor"))
    print(f"done — saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
