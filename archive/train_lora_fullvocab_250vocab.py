"""ARCHIVED SNAPSHOT — do not run. Preserved for traceability only.

This is the exact code that trained the 250-word-vocab checkpoint (the model
behind the "82.7% -> 95.2%" figures reported early in the project). It was
later edited in place (see the active research/uxtd_transcriber/train_lora_fullvocab.py)
to add a --seed flag and combine in the v2/v3 vocabulary expansion (250 -> 491
words), which is what produced the current "80.0% -> 96.0%" figures. That edit
happened before this repo's first commit, so git history doesn't preserve the
pre-edit version — this file is a manual reconstruction of that state, saved
here so the code behind the superseded numbers stays inspectable.

Original docstring, preserved as-is below:
---
LoRA fine-tune on UXTD (clean) + all 250 real StoryWeaver product words —
supersedes the earlier 100-sentence-only augmentation (this set includes
those same 100 plus the 150 Level 1-3 words too), directly closing the
vocabulary-mismatch gap: UXTD's word lists are different words than ours.
"""
import csv

import torch.distributed.tensor  # noqa: F401 — must import before peft, see lessons.md
from peft import LoraConfig, get_peft_model
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration, WhisperProcessor

from train import DATA_DIR, MODEL_NAME, UXTDDataset, WhisperDataCollator

OUTPUT_DIR_FULLVOCAB = __import__("pathlib").Path(__file__).parent / "uxtd_whisper_small_lora_fullvocab"


def build_combined_train_csv() -> None:
    with open(DATA_DIR / "train.csv") as f:
        uxtd_rows = list(csv.DictReader(f))
    with open(DATA_DIR / "augment_full_vocab.csv") as f:
        augment_rows = list(csv.DictReader(f))

    combined = uxtd_rows + augment_rows
    with open(DATA_DIR / "train_fullvocab.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["wav_path", "text", "speaker"])
        writer.writeheader()
        writer.writerows(combined)
    print(f"combined train set: {len(uxtd_rows)} UXTD + {len(augment_rows)} synthetic (full vocab) = {len(combined)} total")


def main() -> None:
    build_combined_train_csv()

    processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="en", task="transcribe")
    base_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    base_model.generation_config.language = "en"
    base_model.generation_config.task = "transcribe"

    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05)
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    train_dataset = UXTDDataset(DATA_DIR / "train_fullvocab.csv", processor)
    val_dataset = UXTDDataset(DATA_DIR / "val.csv", processor)
    print(f"train: {len(train_dataset)}, val: {len(val_dataset)}")

    data_collator = WhisperDataCollator(processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR_FULLVOCAB),
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        num_train_epochs=3,
        eval_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=False,
        fp16=True,
        logging_steps=25,
        report_to=[],
        label_names=["labels"],
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
    model.save_pretrained(str(OUTPUT_DIR_FULLVOCAB / "final"))
    processor.save_pretrained(str(OUTPUT_DIR_FULLVOCAB / "final"))
    print(f"done — saved to {OUTPUT_DIR_FULLVOCAB / 'final'}")


if __name__ == "__main__":
    main()
