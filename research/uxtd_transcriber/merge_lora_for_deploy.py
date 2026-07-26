"""Merges the LoRA adapter into base Whisper-small weights, producing a
standalone HF model directory — required before CTranslate2 conversion
(ct2-transformers-converter needs a plain merged model, not a PEFT adapter)."""
from pathlib import Path

from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from train import MODEL_NAME

LORA_DIR = "/mnt/c/Users/rosha/Documents/sw_2/research/uxtd_transcriber/uxtd_whisper_small_lora_fullvocab/final"
MERGED_DIR = Path(__file__).parent / "uxtd_whisper_small_merged"


def main():
    print("loading base model + LoRA adapter...")
    base_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model = PeftModel.from_pretrained(base_model, LORA_DIR)

    print("merging LoRA weights into base...")
    merged = model.merge_and_unload()

    processor = WhisperProcessor.from_pretrained(LORA_DIR)

    MERGED_DIR.mkdir(exist_ok=True)
    merged.save_pretrained(str(MERGED_DIR))
    processor.save_pretrained(str(MERGED_DIR))
    print(f"done — merged model saved to {MERGED_DIR}")


if __name__ == "__main__":
    main()
