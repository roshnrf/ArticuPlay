"""Full fine-tune of Whisper-small on UXTD (58 typically-developing child
speakers, 3640 train / 428 val / 544 test utterances, speaker-disjoint split).

Full fine-tune (not LoRA) here — unlike Phase 1's 864-example case, 3640
training utterances across 45 speakers is enough data to support it, and full
fine-tuning has more capacity to adapt than LoRA when the data supports it.

Follows the standard HuggingFace Whisper fine-tuning pattern (Seq2SeqTrainer +
data collator with label padding) rather than a hand-rolled loop — this is a
well-established, heavily-used recipe, not something to reinvent.
"""
import csv
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

DATA_DIR = Path(__file__).parent / "data"
WHISPER_SAMPLE_RATE = 16000
MODEL_NAME = "openai/whisper-small"
OUTPUT_DIR = Path(__file__).parent / "uxtd_whisper_small"


def load_audio_resampled(path: str) -> np.ndarray:
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
    return waveform.squeeze(0).numpy()


class UXTDDataset(Dataset):
    def __init__(self, csv_path: Path, processor: WhisperProcessor):
        with open(csv_path) as f:
            self.rows = list(csv.DictReader(f))
        self.processor = processor

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        audio = load_audio_resampled(str(DATA_DIR / row["wav_path"]))
        input_features = self.processor.feature_extractor(
            audio, sampling_rate=WHISPER_SAMPLE_RATE
        ).input_features[0]
        labels = self.processor.tokenizer(row["text"]).input_ids
        return {"input_features": input_features, "labels": labels}


@dataclass
class WhisperDataCollator:
    processor: WhisperProcessor

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        return batch


def main() -> None:
    processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="en", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.generation_config.language = "en"
    model.generation_config.task = "transcribe"

    train_dataset = UXTDDataset(DATA_DIR / "train.csv", processor)
    val_dataset = UXTDDataset(DATA_DIR / "val.csv", processor)
    print(f"train: {len(train_dataset)}, val: {len(val_dataset)}")

    data_collator = WhisperDataCollator(processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        num_train_epochs=3,
        eval_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        logging_steps=25,
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=processor,
    )

    trainer.train()
    trainer.save_model(str(OUTPUT_DIR / "final"))
    processor.save_pretrained(str(OUTPUT_DIR / "final"))
    print(f"done — model saved to {OUTPUT_DIR / 'final'}")


if __name__ == "__main__":
    main()
