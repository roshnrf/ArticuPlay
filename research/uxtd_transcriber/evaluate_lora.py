"""Same WER methodology as evaluate.py, applied to the clean-data LoRA model."""
import csv

import jiwer
import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from train import DATA_DIR, MODEL_NAME, WHISPER_SAMPLE_RATE, load_audio_resampled

device = "cuda"
LORA_DIR = "/mnt/c/Users/rosha/Documents/sw_2/research/uxtd_transcriber/uxtd_whisper_small_lora/final"


def transcribe_all(model, processor, rows):
    model.eval()
    predictions = []
    with torch.no_grad():
        for row in rows:
            audio = load_audio_resampled(str(DATA_DIR / row["wav_path"]))
            inputs = processor.feature_extractor(audio, sampling_rate=WHISPER_SAMPLE_RATE, return_tensors="pt")
            generated_ids = model.generate(
                inputs.input_features.to(device), language="en", task="transcribe", max_new_tokens=64
            )
            text = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            predictions.append(text.strip())
    return predictions


def main():
    with open(DATA_DIR / "test.csv") as f:
        test_rows = list(csv.DictReader(f))
    references = [r["text"] for r in test_rows]
    print(f"evaluating on {len(test_rows)} held-out test utterances (clean data, LoRA)")

    base_processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="en", task="transcribe")
    base_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    base_predictions = transcribe_all(base_model, base_processor, test_rows)
    base_wer = jiwer.wer(references, base_predictions)
    print(f"zero-shot WER = {base_wer:.3f}")

    lora_processor = WhisperProcessor.from_pretrained(LORA_DIR)
    lora_base = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    lora_model = PeftModel.from_pretrained(lora_base, LORA_DIR).to(device)
    lora_model.generation_config = lora_base.generation_config
    lora_predictions = transcribe_all(lora_model, lora_processor, test_rows)
    lora_wer = jiwer.wer(references, lora_predictions)
    print(f"LoRA fine-tuned WER = {lora_wer:.3f}")

    print(f"\nWER improvement: {base_wer - lora_wer:+.3f}")
    print(f"relative reduction: {(base_wer - lora_wer) / base_wer * 100:.1f}%")


if __name__ == "__main__":
    main()
