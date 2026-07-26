"""Real deliverable for 2a: does fine-tuning actually improve child-voice
transcription, measured by Word Error Rate on the held-out test set (7 speakers
never seen during training or validation)? Compares the fine-tuned model
against zero-shot pretrained Whisper-small on the exact same test utterances."""
import csv

import jiwer
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from train import DATA_DIR, OUTPUT_DIR, WHISPER_SAMPLE_RATE, load_audio_resampled

device = "cuda" if torch.cuda.is_available() else "cpu"


def transcribe_all(model, processor, rows: list[dict]) -> list[str]:
    model.eval()
    predictions = []
    with torch.no_grad():
        for row in rows:
            audio = load_audio_resampled(str(DATA_DIR / row["wav_path"]))
            inputs = processor.feature_extractor(audio, sampling_rate=WHISPER_SAMPLE_RATE, return_tensors="pt")
            input_features = inputs.input_features.to(device)
            generated_ids = model.generate(input_features, language="en", task="transcribe", max_new_tokens=64)
            text = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            predictions.append(text.strip())
    return predictions


def main() -> None:
    with open(DATA_DIR / "test.csv") as f:
        test_rows = list(csv.DictReader(f))
    references = [r["text"] for r in test_rows]

    print(f"evaluating on {len(test_rows)} held-out test utterances (speakers never seen in train/val)")

    print("\n--- zero-shot pretrained Whisper-small ---")
    base_processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="en", task="transcribe")
    base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)
    base_predictions = transcribe_all(base_model, base_processor, test_rows)
    base_wer = jiwer.wer(references, base_predictions)
    print(f"zero-shot WER = {base_wer:.3f}")

    del base_model
    torch.cuda.empty_cache()

    print("\n--- fine-tuned on UXTD ---")
    ft_processor = WhisperProcessor.from_pretrained(str(OUTPUT_DIR / "final"))
    ft_model = WhisperForConditionalGeneration.from_pretrained(str(OUTPUT_DIR / "final")).to(device)
    ft_predictions = transcribe_all(ft_model, ft_processor, test_rows)
    ft_wer = jiwer.wer(references, ft_predictions)
    print(f"fine-tuned WER = {ft_wer:.3f}")

    print(f"\nWER improvement: {base_wer - ft_wer:+.3f} (lower WER is better)")
    print(f"relative reduction: {(base_wer - ft_wer) / base_wer * 100:.1f}%")

    print("\n--- sample comparisons (first 5) ---")
    for i in range(min(5, len(test_rows))):
        print(f"reference:  {references[i]!r}")
        print(f"zero-shot:  {base_predictions[i]!r}")
        print(f"fine-tuned: {ft_predictions[i]!r}")
        print()


if __name__ == "__main__":
    main()
