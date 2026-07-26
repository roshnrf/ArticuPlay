"""LoRA fine-tune on UXTD (clean) + synthetic sentence augmentation from
StoryWeaver's own Level 4/5 content — targets the diagnosed gap (no
grammatical-sentence-structure exposure) directly. Val/test stay pure UXTD,
unchanged from the previous run, so results are directly comparable.
"""
import csv

import torch.distributed.tensor  # noqa: F401 — must import before peft, see lessons.md
from peft import LoraConfig, get_peft_model
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration, WhisperProcessor

from train import DATA_DIR, MODEL_NAME, UXTDDataset, WhisperDataCollator

OUTPUT_DIR_AUGMENTED = __import__("pathlib").Path(__file__).parent / "uxtd_whisper_small_lora_augmented"


def build_combined_train_csv() -> None:
    with open(DATA_DIR / "train.csv") as f:
        uxtd_rows = list(csv.DictReader(f))
    with open(DATA_DIR / "augment_sentences.csv") as f:
        augment_rows = list(csv.DictReader(f))

    combined = uxtd_rows + augment_rows
    with open(DATA_DIR / "train_augmented.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["wav_path", "text", "speaker"])
        writer.writeheader()
        writer.writerows(combined)
    print(f"combined train set: {len(uxtd_rows)} UXTD + {len(augment_rows)} synthetic = {len(combined)} total")


def main() -> None:
    build_combined_train_csv()

    processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="en", task="transcribe")
    base_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    base_model.generation_config.language = "en"
    base_model.generation_config.task = "transcribe"

    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05)
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    train_dataset = UXTDDataset(DATA_DIR / "train_augmented.csv", processor)
    val_dataset = UXTDDataset(DATA_DIR / "val.csv", processor)
    print(f"train: {len(train_dataset)}, val: {len(val_dataset)}")

    data_collator = WhisperDataCollator(processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR_AUGMENTED),
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
    model.save_pretrained(str(OUTPUT_DIR_AUGMENTED / "final"))
    processor.save_pretrained(str(OUTPUT_DIR_AUGMENTED / "final"))
    print(f"done — saved to {OUTPUT_DIR_AUGMENTED / 'final'}")


if __name__ == "__main__":
    main()
