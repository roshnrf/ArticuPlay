"""ARCHIVED SNAPSHOT — do not run. Preserved for traceability only.

This is the exact code (word list, model path, docstring) that produced the
"82.7% -> 95.2%" figures reported early in the project (n=50 words, tested
against the 250-word-vocab checkpoint — see train_lora_fullvocab_250vocab.py
in this same archive directory). Once the vocabulary was expanded to 491
words, these 50 words got folded into the trainable set (see
tasks/lessons.md, 2026-07-26 vocab expansion entry), so they no longer
qualified as a held-out test — the active script
research/uxtd_transcriber/evaluate_new_words_heldout.py was edited in place
to a fresh 30-word set instead, which is what produces the current
"80.0% -> 96.0%" figures. That edit happened before this repo's first
commit, so git history doesn't preserve the pre-edit version — this file is
a manual reconstruction, saved so the code behind the superseded numbers
stays inspectable and the two result sets are traceable to distinct code.

Original docstring, preserved as-is below:
---
Genuine held-out generalization test: 50 brand-new real words just added
to the product word bank (backend/scripts/seed_word_content_v2.py) that the
fullvocab LoRA model has NEVER seen in training. Unlike prior evals, this
isn't a re-test on training vocabulary — these words didn't exist when the
model was trained. Uses real compare_ipa phoneme scoring (not substring
exact-match), matching _rescore_with_compare_ipa.py's rigor.
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
AUDIO_DIR = Path("/tmp/heldout_new_words")
device = "cuda"

NEW_WORDS = {
    1: ["mug", "hen", "pot", "jar", "key", "doll", "ring", "sled", "crib", "bell"],
    2: ["carrot", "hairbrush", "lizard", "bunny", "pretzel", "whistle", "ribbon", "needle", "purple", "garlic"],
    3: ["xylophone", "microphone", "ambulance", "caravan", "symphony", "energy", "interest", "beautiful", "important", "newspaper"],
    4: [
        "little yellow duck", "big blue whale", "soft pink pillow", "three red apples", "the happy puppy",
        "her tall brother", "his new shoes", "a cold drink", "the warm sun", "our favorite game",
    ],
    5: [
        "I love my mom.", "She has a big smile.", "He plays with blocks.", "We go to the park.",
        "The girl has a doll.", "My cat likes milk.", "I can read a book.", "She draws a flower.",
        "He jumps very high.", "We sing a song together.",
    ],
}


def synthesize(text: str, out_path: Path) -> None:
    body = json.dumps({"word": text, "language": "en"}).encode()
    req = urllib.request.Request(
        f"{BACKEND}/session/tts", data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        out_path.write_bytes(resp.read())


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
