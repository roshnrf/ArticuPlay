"""Reusable evaluation: base Whisper vs the fine-tuned LoRA model, against
ALL 250 real StoryWeaver product words — deliberately full-scale, not a small
spot-check sample, since a small sample gave a falsely optimistic picture
earlier in this project (see tasks/lessons.md). Uses real audio generated via
the actual /session/tts endpoint, and beam search decoding by default (found
to fix most premature-truncation failures at zero retraining cost).

Re-run this after any future retrain before trusting a held-out-only metric.
"""
import json
import time
import urllib.request

import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from train import MODEL_NAME, WHISPER_SAMPLE_RATE, load_audio_resampled

BACKEND = "http://127.0.0.1:8000/api/v1"
LORA_DIR = "/mnt/c/Users/rosha/Documents/sw_2/research/uxtd_transcriber/uxtd_whisper_small_lora_fullvocab/final"
device = "cuda"


def fetch_all_words() -> list[dict]:
    words = []
    for level in range(1, 6):
        req = urllib.request.Request(f"{BACKEND}/content/words/{level}?lang=en")
        with urllib.request.urlopen(req) as resp:
            level_words = json.loads(resp.read())
            for w in level_words:
                w["level"] = level
            words.extend(level_words)
    return words


def synthesize(word: str, out_path: str) -> None:
    body = json.dumps({"word": word, "language": "en"}).encode()
    req = urllib.request.Request(
        f"{BACKEND}/session/tts", data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        with open(out_path, "wb") as f:
            f.write(resp.read())


def transcribe(model, processor, audio, target_word, use_beam=False):
    inputs = processor.feature_extractor(audio, sampling_rate=WHISPER_SAMPLE_RATE, return_tensors="pt")
    input_features = inputs.input_features.to(device)
    prompt_ids = processor.get_prompt_ids(target_word, return_tensors="pt").to(device)
    kwargs = dict(language="en", task="transcribe", prompt_ids=prompt_ids, max_new_tokens=32)
    if use_beam:
        kwargs.update(num_beams=5, length_penalty=1.5)
    generated_ids = model.generate(input_features, **kwargs)
    return processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()


def main():
    import os

    print("fetching all 250 real product words from the backend...")
    words = fetch_all_words()
    print(f"got {len(words)} words across levels 1-5")

    if os.path.exists(f"/tmp/full_{len(words)-1}.wav"):
        print("reusing existing audio files from the previous test run (same words, same TTS)")
    else:
        print("synthesizing audio for all words via real /session/tts...")
        t0 = time.time()
        for i, w in enumerate(words):
            synthesize(w["word"], f"/tmp/full_{i}.mp3")
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(words)} synthesized")
        print(f"synthesis done in {time.time() - t0:.1f}s")

        import subprocess

        for i in range(len(words)):
            subprocess.run(
                ["ffmpeg", "-y", "-i", f"/tmp/full_{i}.mp3", "-ar", "16000", "-ac", "1", f"/tmp/full_{i}.wav", "-loglevel", "error"],
                check=True,
            )
        print("conversion done")

    print("loading base pretrained Whisper-small...")
    base_processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="en", task="transcribe")
    base_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)

    print("loading clean-data LoRA fine-tuned model...")
    lora_processor = WhisperProcessor.from_pretrained(LORA_DIR)
    lora_base = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    lora_model = PeftModel.from_pretrained(lora_base, LORA_DIR).to(device)
    lora_model.generation_config = lora_base.generation_config

    results = []
    base_correct = 0
    lora_correct = 0
    per_level = {}

    for i, w in enumerate(words):
        audio = load_audio_resampled(f"/tmp/full_{i}.wav")
        base_text = transcribe(base_model, base_processor, audio, w["word"], use_beam=True)
        lora_text = transcribe(lora_model, lora_processor, audio, w["word"], use_beam=True)
        base_ok = w["word"].lower() in base_text.lower()
        lora_ok = w["word"].lower() in lora_text.lower()
        base_correct += base_ok
        lora_correct += lora_ok

        lvl = w["level"]
        per_level.setdefault(lvl, {"base": 0, "lora": 0, "n": 0})
        per_level[lvl]["base"] += base_ok
        per_level[lvl]["lora"] += lora_ok
        per_level[lvl]["n"] += 1

        results.append({"word": w["word"], "level": lvl, "base_text": base_text, "lora_text": lora_text, "base_ok": base_ok, "lora_ok": lora_ok})

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(words)} tested — running: base {base_correct}/{i+1}, lora {lora_correct}/{i+1}")

    with open("/tmp/full_prodword_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== FINAL: base {base_correct}/{len(words)} ({base_correct/len(words)*100:.1f}%)   lora {lora_correct}/{len(words)} ({lora_correct/len(words)*100:.1f}%) ===")
    print("\nper level:")
    for lvl in sorted(per_level):
        d = per_level[lvl]
        print(f"  level {lvl}: base {d['base']}/{d['n']} ({d['base']/d['n']*100:.0f}%)   lora {d['lora']}/{d['n']} ({d['lora']/d['n']*100:.0f}%)")

    print("\nwords where LoRA beat base:")
    for r in results:
        if r["lora_ok"] and not r["base_ok"]:
            print(f"  {r['word']!r}: base={r['base_text']!r} lora={r['lora_text']!r}")

    print("\nwords where base beat LoRA:")
    for r in results:
        if r["base_ok"] and not r["lora_ok"]:
            print(f"  {r['word']!r}: base={r['base_text']!r} lora={r['lora_text']!r}")


if __name__ == "__main__":
    main()
