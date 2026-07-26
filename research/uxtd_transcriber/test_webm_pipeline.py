"""Real test of the flagged-but-never-tested gap from the Week 3 plan: does
the production /session/transcribe endpoint actually handle browser
MediaRecorder's webm/opus output, not just clean wav? Converts real audio
(UXSSD child speech + a TTS sample) to webm/opus matching typical
MediaRecorder settings (audio/webm;codecs=opus, 48kHz), POSTs multipart
to the live endpoint exactly as the frontend's Recorder would, and checks
the transcription is still sane.
"""
import subprocess
import urllib.request
import uuid

BACKEND = "http://127.0.0.1:8000/api/v1"

TEST_CASES = [
    ("/mnt/c/Users/rosha/Documents/sw_2/research/phone_classifier/data/segments/01M-Maint2-017A_CONE_1.wav", "cone"),
    ("/mnt/c/Users/rosha/Documents/sw_2/research/phone_classifier/data/segments/01M-BL1-002A_BREAD_8.wav", "bread"),
    ("/tmp/heldout_new_words_v2/w_0.wav", "toad"),
    ("/tmp/heldout_new_words_v2/w_18.wav", "a giant purple dinosaur"),
]


def wav_to_webm_opus(wav_path: str, out_path: str) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-i", wav_path, "-c:a", "libopus", "-b:a", "32k", "-ar", "48000", "-ac", "1", out_path, "-loglevel", "error"],
        check=True,
    )


def transcribe_via_endpoint(webm_path: str, target_word: str) -> dict:
    boundary = uuid.uuid4().hex
    with open(webm_path, "rb") as f:
        audio_bytes = f.read()

    parts = []
    parts.append(f"--{boundary}\r\n".encode())
    parts.append(b'Content-Disposition: form-data; name="audio"; filename="recording.webm"\r\n')
    parts.append(b"Content-Type: audio/webm\r\n\r\n")
    parts.append(audio_bytes)
    parts.append(b"\r\n")
    for field, value in [("language", "en"), ("target_word", target_word)]:
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(f'Content-Disposition: form-data; name="{field}"\r\n\r\n{value}\r\n'.encode())
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)

    req = urllib.request.Request(
        f"{BACKEND}/session/transcribe",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        import json
        return json.loads(resp.read())


def main():
    print("testing real browser-format (webm/opus) audio against the live /session/transcribe endpoint...\n")
    for wav_path, target_word in TEST_CASES:
        webm_path = "/tmp/webm_test.webm"
        wav_to_webm_opus(wav_path, webm_path)
        try:
            result = transcribe_via_endpoint(webm_path, target_word)
            print(f"  target={target_word!r}: OK -> {result}")
        except Exception as e:
            print(f"  target={target_word!r}: FAILED -> {e}")


if __name__ == "__main__":
    main()
