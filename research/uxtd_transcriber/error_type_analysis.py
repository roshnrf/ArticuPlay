"""False-accept / false-reject breakdown using REAL disordered child speech
(UXSSD corpus, clinician-labeled: primary_score>=4 = correct pronunciation,
<4 = genuine articulation error) — not synthetic TTS. This is the only
dataset available with ground-truth "this pronunciation is actually wrong"
labels, so it's the only way to measure false-accept honestly.

label=1 (real correct pronunciation) -> false-reject if model scores <0.8
label=0 (real disordered pronunciation) -> false-accept if model scores >=0.8
"""
import csv
import random
import sys

import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from train import MODEL_NAME, WHISPER_SAMPLE_RATE, load_audio_resampled

sys.path.insert(0, "/mnt/c/Users/rosha/Documents/sw_2/backend")
from app.utils.compare_ipa import compare_ipa
from app.utils.ipa import to_ipa

LORA_DIR = "/mnt/c/Users/rosha/Documents/sw_2/research/uxtd_transcriber/uxtd_whisper_small_lora_fullvocab/final"
PHONE_CLASSIFIER_DATA = "/mnt/c/Users/rosha/Documents/sw_2/research/phone_classifier/data"
PASS_THRESHOLD = 0.8
N_PER_CLASS = 100
device = "cuda"

random.seed(42)


def transcribe(model, processor, audio, target_word):
    inputs = processor.feature_extractor(audio, sampling_rate=WHISPER_SAMPLE_RATE, return_tensors="pt")
    input_features = inputs.input_features.to(device)
    prompt_ids = processor.get_prompt_ids(target_word, return_tensors="pt").to(device)
    generated_ids = model.generate(
        input_features, language="en", task="transcribe", prompt_ids=prompt_ids,
        max_new_tokens=32, num_beams=5, length_penalty=1.5,
    )
    return processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()


def main():
    with open(f"{PHONE_CLASSIFIER_DATA}/examples.csv") as f:
        rows = list(csv.DictReader(f))

    correct_rows = [r for r in rows if r["label"] == "1"]
    incorrect_rows = [r for r in rows if r["label"] == "0"]
    random.shuffle(correct_rows)
    random.shuffle(incorrect_rows)
    sample = correct_rows[:N_PER_CLASS] + incorrect_rows[:N_PER_CLASS]
    print(f"testing {len(correct_rows[:N_PER_CLASS])} real-correct + {len(incorrect_rows[:N_PER_CLASS])} real-disordered UXSSD segments")

    print("loading fullvocab LoRA model...")
    processor = WhisperProcessor.from_pretrained(LORA_DIR)
    base_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    model = PeftModel.from_pretrained(base_model, LORA_DIR).to(device)
    model.generation_config = base_model.generation_config

    false_rejects, false_accepts = [], []
    true_accepts, true_rejects = 0, 0

    for i, row in enumerate(sample):
        wav_path = f"{PHONE_CLASSIFIER_DATA}/{row['segment_path']}"
        target_word = row["word"].lower()
        audio = load_audio_resampled(wav_path)
        hyp_text = transcribe(model, processor, audio, target_word)

        target_ipa = to_ipa(target_word, language="en")
        hyp_ipa = to_ipa(hyp_text, language="en")
        score = compare_ipa(target_ipa, hyp_ipa).accuracy
        model_pass = score >= PASS_THRESHOLD
        is_really_correct = row["label"] == "1"

        if is_really_correct and not model_pass:
            false_rejects.append({"word": target_word, "score": score, "hyp": hyp_text, "utt": row["utt"]})
        elif is_really_correct and model_pass:
            true_accepts += 1
        elif not is_really_correct and model_pass:
            false_accepts.append({"word": target_word, "score": score, "hyp": hyp_text, "utt": row["utt"], "phone": row["phone"]})
        else:
            true_rejects += 1

        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(sample)}")

    n_correct = len(correct_rows[:N_PER_CLASS])
    n_incorrect = len(incorrect_rows[:N_PER_CLASS])
    print(f"\n=== ERROR-TYPE ANALYSIS (real UXSSD child speech, n={len(sample)}) ===")
    print(f"real-correct pronunciations (n={n_correct}): true-accept {true_accepts} ({true_accepts/n_correct*100:.1f}%)   false-reject {len(false_rejects)} ({len(false_rejects)/n_correct*100:.1f}%)")
    print(f"real-disordered pronunciations (n={n_incorrect}): true-reject {true_rejects} ({true_rejects/n_incorrect*100:.1f}%)   false-accept {len(false_accepts)} ({len(false_accepts)/n_incorrect*100:.1f}%)")

    print(f"\nfalse-accept examples (model passed a genuinely wrong pronunciation):")
    for e in false_accepts[:10]:
        print(f"  {e['word']!r} (bad {e['phone']}): score={e['score']:.2f} hyp={e['hyp']!r}")

    print(f"\nfalse-reject examples (model failed a genuinely correct pronunciation):")
    for e in false_rejects[:10]:
        print(f"  {e['word']!r}: score={e['score']:.2f} hyp={e['hyp']!r}")


if __name__ == "__main__":
    main()
