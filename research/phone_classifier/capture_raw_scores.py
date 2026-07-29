"""Captures raw continuous compare_ipa accuracy + phone_classifier_flag for the
same 200 real UXSSD samples used throughout this investigation — ONE inference
pass, saved to disk, so PASS_THRESHOLD can be swept offline afterward without
re-running the model 5x. Uses the live /session/transcribe endpoint (real
pipeline), not the raw model directly.
"""
import csv
import json
import random
import uuid

from test_integration_e2e import make_real_session, multipart_transcribe

PHONE_CLASSIFIER_DATA = "/mnt/c/Users/rosha/Documents/sw_2/research/phone_classifier/data"
N_PER_CLASS = 100

random.seed(42)


def main():
    with open(f"{PHONE_CLASSIFIER_DATA}/examples.csv") as f:
        rows = list(csv.DictReader(f))
    correct_rows = [r for r in rows if r["label"] == "1"]
    incorrect_rows = [r for r in rows if r["label"] == "0"]
    random.shuffle(correct_rows)
    random.shuffle(incorrect_rows)
    sample = correct_rows[:N_PER_CLASS] + incorrect_rows[:N_PER_CLASS]
    print(f"capturing raw scores for {len(sample)} real UXSSD samples...")

    import sys
    sys.path.insert(0, "/mnt/c/Users/rosha/Documents/sw_2/backend")
    from app.utils.compare_ipa import compare_ipa
    from app.utils.ipa import to_ipa

    results = []
    for i, row in enumerate(sample):
        wav_path = f"{PHONE_CLASSIFIER_DATA}/{row['segment_path']}"
        target_word = row["word"].lower()
        asr_result = multipart_transcribe(wav_path, target_word)
        transcript = asr_result["transcript"]
        phone_flag = asr_result.get("phone_classifier_flag")

        target_ipa = to_ipa(target_word, language="en")
        hyp_ipa = to_ipa(transcript, language="en")
        accuracy = compare_ipa(target_ipa, hyp_ipa).accuracy

        results.append({
            "word": target_word, "is_really_correct": row["label"] == "1",
            "transcript": transcript, "accuracy": accuracy, "phone_flag": phone_flag,
        })
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(sample)}")

    with open("/tmp/raw_scores_200.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"done — saved to /tmp/raw_scores_200.json")


if __name__ == "__main__":
    main()
