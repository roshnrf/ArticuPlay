"""Tests whether the target-word prompt_ids bias is actually the cause of the
28% false-accept rate found in error_type_analysis.py. Runs the SAME 200 UXSSD
segments through the fullvocab model twice per sample: once with prompt_ids
(current production behavior) and once without (plain beam decode, no hint).
If false-accept drops meaningfully without bias, that confirms the bias is
suppressing real acoustic error signal — and quantifies the word-recognition
cost of removing it (false-reject may also change, check both directions).
"""
import csv
import random
import sys

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


def transcribe(model, processor, audio, target_word=None):
    inputs = processor.feature_extractor(audio, sampling_rate=WHISPER_SAMPLE_RATE, return_tensors="pt")
    input_features = inputs.input_features.to(device)
    kwargs = dict(language="en", task="transcribe", max_new_tokens=32, num_beams=5, length_penalty=1.5)
    if target_word is not None:
        kwargs["prompt_ids"] = processor.get_prompt_ids(target_word, return_tensors="pt").to(device)
    generated_ids = model.generate(input_features, **kwargs)
    return processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()


def score(target_word, hyp_text):
    target_ipa = to_ipa(target_word, language="en")
    hyp_ipa = to_ipa(hyp_text, language="en")
    return compare_ipa(target_ipa, hyp_ipa).accuracy


def main():
    with open(f"{PHONE_CLASSIFIER_DATA}/examples.csv") as f:
        rows = list(csv.DictReader(f))

    correct_rows = [r for r in rows if r["label"] == "1"]
    incorrect_rows = [r for r in rows if r["label"] == "0"]
    random.shuffle(correct_rows)
    random.shuffle(incorrect_rows)
    sample = correct_rows[:N_PER_CLASS] + incorrect_rows[:N_PER_CLASS]
    n_correct, n_incorrect = len(correct_rows[:N_PER_CLASS]), len(incorrect_rows[:N_PER_CLASS])
    print(f"testing {n_correct} real-correct + {n_incorrect} real-disordered, WITH vs WITHOUT prompt bias")

    processor = WhisperProcessor.from_pretrained(LORA_DIR)
    base_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    model = PeftModel.from_pretrained(base_model, LORA_DIR).to(device)
    model.generation_config = base_model.generation_config

    biased = {"fa": 0, "fr": 0, "ta": 0, "tr": 0}
    unbiased = {"fa": 0, "fr": 0, "ta": 0, "tr": 0}
    examples_unbiased_fa = []
    examples_unbiased_correctly_caught = []

    for i, row in enumerate(sample):
        wav_path = f"{PHONE_CLASSIFIER_DATA}/{row['segment_path']}"
        target_word = row["word"].lower()
        is_really_correct = row["label"] == "1"
        audio = load_audio_resampled(wav_path)

        hyp_biased = transcribe(model, processor, audio, target_word)
        hyp_unbiased = transcribe(model, processor, audio, None)
        score_biased = score(target_word, hyp_biased)
        score_unbiased = score(target_word, hyp_unbiased)
        pass_biased = score_biased >= PASS_THRESHOLD
        pass_unbiased = score_unbiased >= PASS_THRESHOLD

        for tag, is_correct, model_pass, d in [("b", is_really_correct, pass_biased, biased), ("u", is_really_correct, pass_unbiased, unbiased)]:
            if is_correct and not model_pass:
                d["fr"] += 1
            elif is_correct and model_pass:
                d["ta"] += 1
            elif not is_correct and model_pass:
                d["fa"] += 1
            else:
                d["tr"] += 1

        if not is_really_correct and pass_unbiased:
            examples_unbiased_fa.append({"word": target_word, "hyp": hyp_unbiased, "phone": row["phone"]})
        if not is_really_correct and not pass_unbiased and pass_biased:
            examples_unbiased_correctly_caught.append({"word": target_word, "hyp_biased": hyp_biased, "hyp_unbiased": hyp_unbiased, "phone": row["phone"]})

        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(sample)}")

    print(f"\n=== WITH prompt bias (production config) ===")
    print(f"real-correct (n={n_correct}): true-accept {biased['ta']} ({biased['ta']/n_correct*100:.1f}%)   false-reject {biased['fr']} ({biased['fr']/n_correct*100:.1f}%)")
    print(f"real-disordered (n={n_incorrect}): true-reject {biased['tr']} ({biased['tr']/n_incorrect*100:.1f}%)   false-accept {biased['fa']} ({biased['fa']/n_incorrect*100:.1f}%)")

    print(f"\n=== WITHOUT prompt bias (plain beam decode) ===")
    print(f"real-correct (n={n_correct}): true-accept {unbiased['ta']} ({unbiased['ta']/n_correct*100:.1f}%)   false-reject {unbiased['fr']} ({unbiased['fr']/n_correct*100:.1f}%)")
    print(f"real-disordered (n={n_incorrect}): true-reject {unbiased['tr']} ({unbiased['tr']/n_incorrect*100:.1f}%)   false-accept {unbiased['fa']} ({unbiased['fa']/n_incorrect*100:.1f}%)")

    print(f"\ncases where removing bias fixed a false-accept (n={len(examples_unbiased_correctly_caught)}):")
    for e in examples_unbiased_correctly_caught[:10]:
        print(f"  {e['word']!r} (bad {e['phone']}): biased={e['hyp_biased']!r}  unbiased={e['hyp_unbiased']!r}")

    print(f"\nremaining false-accepts even without bias (n={len(examples_unbiased_fa)}):")
    for e in examples_unbiased_fa[:10]:
        print(f"  {e['word']!r} (bad {e['phone']}): unbiased_hyp={e['hyp']!r}")


if __name__ == "__main__":
    main()
