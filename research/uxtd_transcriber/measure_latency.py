"""Real latency measurement for the fullvocab LoRA model + beam search,
the actual inference config used in evaluation — compared against the
product's 3s/item budget (spec doc). Loads the model once, times only
the generate() call (not model load), across 20 real audio samples.
"""
import time

import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from train import MODEL_NAME, WHISPER_SAMPLE_RATE, load_audio_resampled

LORA_DIR = "/mnt/c/Users/rosha/Documents/sw_2/research/uxtd_transcriber/uxtd_whisper_small_lora_fullvocab/final"
AUDIO_DIR = "/tmp/heldout_new_words_v2"
device = "cuda"


def transcribe_timed(model, processor, audio, target_word):
    inputs = processor.feature_extractor(audio, sampling_rate=WHISPER_SAMPLE_RATE, return_tensors="pt")
    input_features = inputs.input_features.to(device)
    prompt_ids = processor.get_prompt_ids(target_word, return_tensors="pt").to(device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    generated_ids = model.generate(
        input_features, language="en", task="transcribe", prompt_ids=prompt_ids,
        max_new_tokens=48, num_beams=5, length_penalty=1.5,
    )
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    text = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    return text, elapsed_ms


def main():
    print("loading fullvocab LoRA model...")
    processor = WhisperProcessor.from_pretrained(LORA_DIR)
    base_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    model = PeftModel.from_pretrained(base_model, LORA_DIR).to(device)
    model.generation_config = base_model.generation_config

    # index must match NEW_WORDS iteration order in evaluate_new_words_heldout.py
    # (level1 x6=idx0-5, level2 x6=idx6-11, level3 x6=idx12-17, level4 x6=idx18-23, level5 x6=idx24-29)
    words = [
        ("toad", 1, 0), ("moth", 1, 1), ("wasp", 1, 2), ("clam", 1, 3), ("shrimp", 1, 4), ("yak", 1, 5),
        ("gerbil", 2, 6), ("tadpole", 2, 7), ("peacock", 2, 8), ("raccoon", 2, 9), ("beaver", 2, 10), ("badger", 2, 11),
        ("electric", 3, 12), ("fantastic", 3, 13), ("wonderful", 3, 14), ("fabulous", 3, 15), ("marvelous", 3, 16), ("athletic", 3, 17),
        ("a giant purple dinosaur", 4, 18), ("the brave little astronaut", 4, 19),
        ("The frog jumped into the pond.", 5, 24), ("She painted a picture of the sky.", 5, 25),
    ]

    print("warming up (first call includes CUDA kernel compilation)...")
    warm_audio = load_audio_resampled(f"{AUDIO_DIR}/w_0.wav")
    transcribe_timed(model, processor, warm_audio, words[0][0])

    latencies = []
    for word, level, idx in words:
        audio = load_audio_resampled(f"{AUDIO_DIR}/w_{idx}.wav")
        text, ms = transcribe_timed(model, processor, audio, word)
        latencies.append((word, level, ms))
        print(f"  {word!r} (level {level}): {ms:.0f}ms  -> {text!r}")

    ms_values = sorted(ms for _, _, ms in latencies)
    n = len(ms_values)
    mean_ms = sum(ms_values) / n
    p95_ms = ms_values[int(n * 0.95)] if n > 1 else ms_values[0]
    max_ms = ms_values[-1]

    print(f"\n=== LATENCY (fullvocab LoRA, beam=5, GPU, n={n}) ===")
    print(f"mean: {mean_ms:.0f}ms   p95: {p95_ms:.0f}ms   max: {max_ms:.0f}ms")
    print(f"budget: 3000ms/item — {'PASS' if max_ms < 3000 else 'FAIL'} (max case)")
    print("\nNote: this is model inference only (GPU). Real deployment adds network")
    print("upload of the audio blob + FastAPI request overhead + (if CPU-only host)")
    print("a likely much slower forward pass — GPU inference time is a lower bound,")
    print("not the full user-perceived latency.")


if __name__ == "__main__":
    main()
