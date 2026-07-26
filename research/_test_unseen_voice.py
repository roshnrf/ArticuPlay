"""Real generalization check: tests both models against a TTS voice
(en-US-GuyNeural) that neither ever trained OR was tested on before —
everything so far used en-US-AnaNeural exclusively. If the fine-tuned model
still wins here, that's evidence of real word/structure learning, not just
memorizing one specific voice's acoustic signature.
"""
import asyncio
import json
import subprocess

import edge_tts
import jiwer
import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import sys
import urllib.request

sys.path.insert(0, "/mnt/c/Users/rosha/Documents/sw_2/research/uxtd_transcriber")
from train import load_audio_resampled, WHISPER_SAMPLE_RATE, MODEL_NAME

device = "cuda"
LORA_DIR = "/mnt/c/Users/rosha/Documents/sw_2/research/uxtd_transcriber/uxtd_whisper_small_lora_fullvocab/final"
BACKEND = "http://127.0.0.1:8000/api/v1"
UNSEEN_VOICE = "en-US-GuyNeural"


def fetch_all_words():
    words = []
    for level in range(1, 6):
        req = urllib.request.Request(f"{BACKEND}/content/words/{level}?lang=en")
        with urllib.request.urlopen(req) as resp:
            for w in json.loads(resp.read()):
                w["level"] = level
                words.append(w)
    return words


async def synth(text, path):
    c = edge_tts.Communicate(text, UNSEEN_VOICE)
    with open(path, "wb") as f:
        async for chunk in c.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])


def transcribe(model, processor, audio, target_word):
    inputs = processor.feature_extractor(audio, sampling_rate=WHISPER_SAMPLE_RATE, return_tensors="pt")
    input_features = inputs.input_features.to(device)
    prompt_ids = processor.get_prompt_ids(target_word, return_tensors="pt").to(device)
    ids = model.generate(input_features, language="en", task="transcribe", prompt_ids=prompt_ids,
                          max_new_tokens=32, num_beams=5, length_penalty=1.5)
    return processor.tokenizer.decode(ids[0], skip_special_tokens=True).strip()


async def main():
    words = fetch_all_words()
    print(f"generating {len(words)} words in UNSEEN voice ({UNSEEN_VOICE})...")
    for i, w in enumerate(words):
        await synth(w["word"], f"/tmp/unseen_{i}.mp3")
        subprocess.run(["ffmpeg", "-y", "-i", f"/tmp/unseen_{i}.mp3", "-ar", "16000", "-ac", "1", f"/tmp/unseen_{i}.wav", "-loglevel", "error"], check=True)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(words)}")

    base_processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="en", task="transcribe")
    base_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)

    lora_processor = WhisperProcessor.from_pretrained(LORA_DIR)
    lora_base = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    lora_model = PeftModel.from_pretrained(lora_base, LORA_DIR).to(device)
    lora_model.generation_config = lora_base.generation_config

    base_correct = 0
    lora_correct = 0
    refs, base_preds, lora_preds = [], [], []
    per_level = {}

    for i, w in enumerate(words):
        audio = load_audio_resampled(f"/tmp/unseen_{i}.wav")
        base_text = transcribe(base_model, base_processor, audio, w["word"])
        lora_text = transcribe(lora_model, lora_processor, audio, w["word"])
        base_ok = w["word"].lower() in base_text.lower()
        lora_ok = w["word"].lower() in lora_text.lower()
        base_correct += base_ok
        lora_correct += lora_ok
        refs.append(w["word"])
        base_preds.append(base_text)
        lora_preds.append(lora_text)

        lvl = w["level"]
        per_level.setdefault(lvl, {"base": 0, "lora": 0, "n": 0})
        per_level[lvl]["base"] += base_ok
        per_level[lvl]["lora"] += lora_ok
        per_level[lvl]["n"] += 1

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(words)} tested")

    print(f"\n=== UNSEEN VOICE TEST ({UNSEEN_VOICE}) ===")
    print(f"base: {base_correct}/{len(words)} ({base_correct/len(words)*100:.1f}%)   lora: {lora_correct}/{len(words)} ({lora_correct/len(words)*100:.1f}%)")
    print(f"base WER = {jiwer.wer(refs, base_preds):.4f}   lora WER = {jiwer.wer(refs, lora_preds):.4f}")
    print("\nper level:")
    for lvl in sorted(per_level):
        d = per_level[lvl]
        print(f"  level {lvl}: base {d['base']}/{d['n']} ({d['base']/d['n']*100:.0f}%)   lora {d['lora']}/{d['n']} ({d['lora']/d['n']*100:.0f}%)")


if __name__ == "__main__":
    asyncio.run(main())
