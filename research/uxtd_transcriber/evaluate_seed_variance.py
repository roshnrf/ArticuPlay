"""Variance check: same fresh 30-word holdout, 3 independently trained
models (seeds 42/43/44), same architecture/data/hyperparams. Reuses audio
already synthesized at /tmp/heldout_new_words_v2 by evaluate_new_words_heldout.py."""
import json
import sys

from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from train import MODEL_NAME, WHISPER_SAMPLE_RATE, load_audio_resampled

sys.path.insert(0, "/mnt/c/Users/rosha/Documents/sw_2/backend")
from app.utils.compare_ipa import compare_ipa
from app.utils.ipa import to_ipa

AUDIO_DIR = "/tmp/heldout_new_words_v2"
device = "cuda"

SEEDS = {
    42: "/mnt/c/Users/rosha/Documents/sw_2/research/uxtd_transcriber/uxtd_whisper_small_lora_fullvocab/final",
    43: "/mnt/c/Users/rosha/Documents/sw_2/research/uxtd_transcriber/uxtd_whisper_small_lora_fullvocab_seed43/final",
    44: "/mnt/c/Users/rosha/Documents/sw_2/research/uxtd_transcriber/uxtd_whisper_small_lora_fullvocab_seed44/final",
}

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


def transcribe(model, processor, audio, target_word):
    inputs = processor.feature_extractor(audio, sampling_rate=WHISPER_SAMPLE_RATE, return_tensors="pt")
    input_features = inputs.input_features.to(device)
    prompt_ids = processor.get_prompt_ids(target_word, return_tensors="pt").to(device)
    generated_ids = model.generate(
        input_features, language="en", task="transcribe", prompt_ids=prompt_ids,
        max_new_tokens=48, num_beams=5, length_penalty=1.5,
    )
    return processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()


def main():
    words = [{"word": w, "level": lvl} for lvl, ws in NEW_WORDS.items() for w in ws]
    results_by_seed = {}

    for seed, lora_dir in SEEDS.items():
        print(f"\nloading seed {seed} model...")
        processor = WhisperProcessor.from_pretrained(lora_dir)
        base_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
        model = PeftModel.from_pretrained(base_model, lora_dir).to(device)
        model.generation_config = base_model.generation_config

        scores = []
        for i, w in enumerate(words):
            audio = load_audio_resampled(f"{AUDIO_DIR}/w_{i}.wav")
            hyp_text = transcribe(model, processor, audio, w["word"])
            target_ipa = to_ipa(w["word"], language="en")
            hyp_ipa = to_ipa(hyp_text, language="en")
            score = compare_ipa(target_ipa, hyp_ipa).accuracy
            scores.append(score)

        mean_acc = sum(scores) / len(scores) * 100
        results_by_seed[seed] = mean_acc
        print(f"seed {seed}: mean phoneme accuracy {mean_acc:.1f}%")
        del model, base_model
        import torch
        torch.cuda.empty_cache()

    vals = list(results_by_seed.values())
    mean_of_means = sum(vals) / len(vals)
    spread = max(vals) - min(vals)
    print(f"\n=== VARIANCE ACROSS SEEDS (n=30 words each, same holdout) ===")
    for seed, acc in results_by_seed.items():
        print(f"  seed {seed}: {acc:.1f}%")
    print(f"mean: {mean_of_means:.1f}%   spread (max-min): {spread:.1f} points")

    with open("/tmp/seed_variance_results.json", "w") as f:
        json.dump(results_by_seed, f, indent=2)


if __name__ == "__main__":
    main()
