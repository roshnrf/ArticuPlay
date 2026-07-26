"""Genuine held-out generalization test v2 — the model now trains on 491 words
(original 250 + v2's 50 + v3's 191), so v2's old 50-word holdout no longer
qualifies as unseen. This is a fresh 30-word set (6/level) that exists ONLY
as a DB content entry, never folded into any training CSV. Uses real
compare_ipa phoneme scoring (not substring exact-match).
"""
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from train import MODEL_NAME, WHISPER_SAMPLE_RATE, load_audio_resampled

sys.path.insert(0, "/mnt/c/Users/rosha/Documents/sw_2/backend")
from app.utils.compare_ipa import compare_ipa
from app.utils.ipa import to_ipa

BACKEND = "http://127.0.0.1:8000/api/v1"
LORA_DIR = "/mnt/c/Users/rosha/Documents/sw_2/research/uxtd_transcriber/uxtd_whisper_small_lora_fullvocab/final"
AUDIO_DIR = Path("/tmp/heldout_new_words_v2")
device = "cuda"

NEW_WORDS = {
    1: ["toad", "moth", "wasp", "clam", "shrimp", "yak"],
    2: ["gerbil", "tadpole", "peacock", "raccoon", "beaver", "badger"],
    3: ["electric", "fantastic", "wonderful", "fabulous", "marvelous", "athletic"],
    4: [
        "a giant purple dinosaur", "the brave little astronaut", "her cozy warm sweater",
        "his favorite blue jacket", "a splashy summer pool", "the quiet library corner",
    ],
    5: [
        "The frog jumped into the pond.", "She painted a picture of the sky.",
        "We ate dinner together as a family.", "He tied his shoes all by himself.",
        "The kids laughed at the silly joke.", "I found my lost toy under the bed.",
    ],
}


def synthesize(text: str, mp3_path: Path) -> None:
    body = json.dumps({"word": text, "language": "en"}).encode()
    req = urllib.request.Request(
        f"{BACKEND}/session/tts", data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        mp3_path.write_bytes(resp.read())


def transcribe(model, processor, audio, target_word, use_beam=True):
    inputs = processor.feature_extractor(audio, sampling_rate=WHISPER_SAMPLE_RATE, return_tensors="pt")
    input_features = inputs.input_features.to(device)
    prompt_ids = processor.get_prompt_ids(target_word, return_tensors="pt").to(device)
    kwargs = dict(language="en", task="transcribe", prompt_ids=prompt_ids, max_new_tokens=48)
    if use_beam:
        kwargs.update(num_beams=5, length_penalty=1.5)
    generated_ids = model.generate(input_features, **kwargs)
    return processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()


def phoneme_accuracy(target_word: str, hyp_text: str) -> float:
    target_ipa = to_ipa(target_word, language="en")
    hyp_ipa = to_ipa(hyp_text, language="en")
    result = compare_ipa(target_ipa, hyp_ipa)
    return result.accuracy


def main():
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    words = [{"word": w, "level": lvl} for lvl, ws in NEW_WORDS.items() for w in ws]
    print(f"held-out generalization test: {len(words)} brand-new words, never in training")

    print("synthesizing audio via real /session/tts...")
    t0 = time.time()
    for i, w in enumerate(words):
        mp3 = AUDIO_DIR / f"w_{i}.mp3"
        wav = AUDIO_DIR / f"w_{i}.wav"
        synthesize(w["word"], mp3)
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(mp3), "-ar", "16000", "-ac", "1", str(wav), "-loglevel", "error"],
            check=True,
        )
    print(f"synthesis done in {time.time() - t0:.1f}s")

    print("loading base pretrained Whisper-small...")
    base_processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="en", task="transcribe")
    base_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)

    print("loading fullvocab LoRA fine-tuned model...")
    lora_processor = WhisperProcessor.from_pretrained(LORA_DIR)
    lora_base = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    lora_model = PeftModel.from_pretrained(lora_base, LORA_DIR).to(device)
    lora_model.generation_config = lora_base.generation_config

    results = []
    base_scores, lora_scores = [], []
    per_level = {}

    for i, w in enumerate(words):
        wav = AUDIO_DIR / f"w_{i}.wav"
        audio = load_audio_resampled(str(wav))
        base_text = transcribe(base_model, base_processor, audio, w["word"])
        lora_text = transcribe(lora_model, lora_processor, audio, w["word"])
        base_acc = phoneme_accuracy(w["word"], base_text)
        lora_acc = phoneme_accuracy(w["word"], lora_text)
        base_scores.append(base_acc)
        lora_scores.append(lora_acc)

        lvl = w["level"]
        per_level.setdefault(lvl, {"base": [], "lora": []})
        per_level[lvl]["base"].append(base_acc)
        per_level[lvl]["lora"].append(lora_acc)

        results.append({
            "word": w["word"], "level": lvl,
            "base_text": base_text, "lora_text": lora_text,
            "base_acc": base_acc, "lora_acc": lora_acc,
        })
        print(f"  [{i+1}/{len(words)}] {w['word']!r}: base={base_acc:.2f} lora={lora_acc:.2f}  (base_text={base_text!r} lora_text={lora_text!r})")

    with open("/tmp/heldout_new_words_results.json", "w") as f:
        json.dump(results, f, indent=2)

    n = len(words)
    print(f"\n=== HELD-OUT GENERALIZATION (brand-new never-trained words, n={n}) ===")
    print(f"base Whisper-small:  mean phoneme accuracy {sum(base_scores)/n*100:.1f}%")
    print(f"fullvocab LoRA:      mean phoneme accuracy {sum(lora_scores)/n*100:.1f}%")
    print("\nper level:")
    for lvl in sorted(per_level):
        d = per_level[lvl]
        nb = len(d["base"])
        print(f"  level {lvl}: base {sum(d['base'])/nb*100:.1f}%   lora {sum(d['lora'])/nb*100:.1f}%")


if __name__ == "__main__":
    main()
