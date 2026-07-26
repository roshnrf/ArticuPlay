"""Retry of the UXTD fine-tune using LoRA instead of full fine-tune — direct
fix for the overfitting/catastrophic-forgetting found when testing the full
fine-tune against real StoryWeaver product words (base Whisper 8/8, full
fine-tune 4/8, including hallucinated leaked-label text). LoRA keeps the base
model's original ~680k-hour pretrained knowledge frozen and only trains small
adapter matrices, the same technique that worked for the Phase 1 phone
classifier — this is the fix for the diagnosed root cause, not a guess.
"""
import torch.distributed.tensor  # noqa: F401 — must import before peft, see lessons.md
from peft import LoraConfig, get_peft_model
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration, WhisperProcessor

from train import DATA_DIR, MODEL_NAME, UXTDDataset, WhisperDataCollator

OUTPUT_DIR_LORA = __import__("pathlib").Path(__file__).parent / "uxtd_whisper_small_lora"


def main() -> None:
    processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="en", task="transcribe")
    base_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    base_model.generation_config.language = "en"
    base_model.generation_config.task = "transcribe"

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # attention layers, both encoder and decoder
        lora_dropout=0.05,
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    train_dataset = UXTDDataset(DATA_DIR / "train.csv", processor)
    val_dataset = UXTDDataset(DATA_DIR / "val.csv", processor)
    print(f"train: {len(train_dataset)}, val: {len(val_dataset)}")

    data_collator = WhisperDataCollator(processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR_LORA),
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,  # LoRA typically wants a higher LR than full fine-tune
        num_train_epochs=3,
        eval_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=False,  # peft-wrapped model needs generate() called differently; skip during training eval
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
    model.save_pretrained(str(OUTPUT_DIR_LORA / "final"))
    processor.save_pretrained(str(OUTPUT_DIR_LORA / "final"))
    print(f"done — LoRA adapter saved to {OUTPUT_DIR_LORA / 'final'}")


if __name__ == "__main__":
    main()
