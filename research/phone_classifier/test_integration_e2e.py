"""The real verification: does the phone-classifier integration actually move
the false-accept/false-reject numbers through the LIVE deployed pipeline
(/session/transcribe -> /session/score), not just the raw model in isolation?
Same 200 real UXSSD samples (100 correct + 100 disordered) as
research/uxtd_transcriber/error_type_analysis.py, so the before/after
comparison is apples-to-apples.
"""
import csv
import json
import random
import urllib.parse
import urllib.request
import uuid

BACKEND = "http://127.0.0.1:8000/api/v1"
PHONE_CLASSIFIER_DATA = "/mnt/c/Users/rosha/Documents/sw_2/research/phone_classifier/data"
N_PER_CLASS = 100

random.seed(42)


def multipart_transcribe(wav_path: str, target_word: str) -> dict:
    boundary = uuid.uuid4().hex
    with open(wav_path, "rb") as f:
        audio_bytes = f.read()
    parts = [
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="audio"; filename="a.wav"\r\n',
        b"Content-Type: audio/wav\r\n\r\n",
        audio_bytes,
        b"\r\n",
    ]
    for field, value in [("language", "en"), ("target_word", target_word)]:
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(f'Content-Disposition: form-data; name="{field}"\r\n\r\n{value}\r\n'.encode())
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    req = urllib.request.Request(
        f"{BACKEND}/session/transcribe", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}, method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def score(session_id: str, item_index: int, target_word: str, child_transcript: str, phone_classifier_flag) -> dict:
    body = json.dumps({
        "session_id": session_id, "item_index": item_index, "target_word": target_word,
        "language": "en", "child_transcript": child_transcript, "attempt_num": 1,
        "phone_classifier_flag": phone_classifier_flag,
    }).encode()
    req = urllib.request.Request(
        f"{BACKEND}/session/score", data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def post_json(path: str, body: dict, token: str | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(f"{BACKEND}{path}", data=json.dumps(body).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def make_real_session() -> str:
    """Registers a throwaway test account through the real API to get a valid
    session_id (score() enforces the FK) — same flow the frontend uses."""
    email = f"phoneclf_test_{uuid.uuid4().hex[:8]}@example.com"
    password = "TestPass123!"
    post_json("/auth/register", {"email": email, "password": password, "full_name": "Test"})

    form_body = urllib.parse.urlencode({"username": email, "password": password}).encode()
    req = urllib.request.Request(
        f"{BACKEND}/auth/login", data=form_body,
        headers={"Content-Type": "application/x-www-form-urlencoded"}, method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        token = json.loads(resp.read())["access_token"]

    child = post_json("/children/", {"name": "Test Child", "age": 6, "language": "en"}, token=token)
    session = post_json("/session/start", {"child_id": child["id"], "language": "en", "level": 1}, token=token)
    return session["id"]


def main():
    with open(f"{PHONE_CLASSIFIER_DATA}/examples.csv") as f:
        rows = list(csv.DictReader(f))
    correct_rows = [r for r in rows if r["label"] == "1"]
    incorrect_rows = [r for r in rows if r["label"] == "0"]
    random.shuffle(correct_rows)
    random.shuffle(incorrect_rows)
    sample = correct_rows[:N_PER_CLASS] + incorrect_rows[:N_PER_CLASS]
    print(f"testing {N_PER_CLASS} real-correct + {N_PER_CLASS} real-disordered through the LIVE pipeline")

    session_id = make_real_session()

    false_accepts, false_rejects, true_accepts, true_rejects = 0, 0, 0, 0
    overrides_that_fixed_fa = 0

    for i, row in enumerate(sample):
        wav_path = f"{PHONE_CLASSIFIER_DATA}/{row['segment_path']}"
        target_word = row["word"].lower()
        is_really_correct = row["label"] == "1"

        asr_result = multipart_transcribe(wav_path, target_word)
        score_result = score(session_id, i, target_word, asr_result["transcript"], asr_result.get("phone_classifier_flag"))
        model_pass = score_result["passed"]

        if is_really_correct and not model_pass:
            false_rejects += 1
        elif is_really_correct and model_pass:
            true_accepts += 1
        elif not is_really_correct and model_pass:
            false_accepts += 1
        else:
            true_rejects += 1
            if score_result.get("phone_classifier_override"):
                overrides_that_fixed_fa += 1

        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(sample)}")

    n = len(sample)
    print(f"\n=== LIVE PIPELINE RESULT (n={n}) ===")
    print(f"real-correct (n={N_PER_CLASS}): true-accept {true_accepts} ({true_accepts/N_PER_CLASS*100:.1f}%)   false-reject {false_rejects} ({false_rejects/N_PER_CLASS*100:.1f}%)")
    print(f"real-disordered (n={N_PER_CLASS}): true-reject {true_rejects} ({true_rejects/N_PER_CLASS*100:.1f}%)   false-accept {false_accepts} ({false_accepts/N_PER_CLASS*100:.1f}%)")
    print(f"of the true-rejects, {overrides_that_fixed_fa} were phone-classifier catches (would've been false-accepts without it)")


if __name__ == "__main__":
    main()
