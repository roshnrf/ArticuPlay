"""ARCHIVED SNAPSHOT — not live code, do not import. Preserved for traceability only.

This is the original production ASRService: stock faster-whisper "base" model,
target-word decoding bias (initial_prompt), beam_size=1. Two changes happened
after this, both driven by real measurements (see tasks/lessons.md):

1. Swapped in the LoRA fine-tuned Whisper (491-word vocab), removed the
   target-word bias, and raised beam_size to 5 — because target-word biasing
   was found to push false-accept to 28% on real disordered speech (the model
   would transcribe back the correct spelling even when the child's actual
   phone was wrong).
2. Added a phone-classifier gate on top of that (see
   backend/app/services/phone_classifier_service.py) — false-accept 28% -> 3%.

Both of those later changes are in the live file's git history
(backend/app/services/asr_service.py) since they happened after this repo's
first commit. This original pre-fine-tune version predates the first commit
entirely, so it isn't recoverable from git log — reconstructed here from
context so the full progression (original -> fine-tuned+unbiased -> +phone
classifier) stays inspectable end to end.
"""
import asyncio
import io
import time
from functools import lru_cache

from faster_whisper import WhisperModel

from app.core.config import settings
from app.schemas.asr import ASRResult


@lru_cache(maxsize=1)
def _get_model() -> WhisperModel:
    """Load the Whisper model once per process. Loading takes ~1-2s; must never happen per-request."""
    return WhisperModel(settings.WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")


class ASRService:
    """Self-hosted speech-to-text via faster-whisper. CPU-bound and sync, so transcribe()
    runs the blocking call in a worker thread to avoid stalling the event loop.

    target_word biases decoding via Whisper's initial_prompt — measured during Week 1 testing:
    on isolated single-word drill audio (no sentence context), base-model accuracy was ~42%
    without it vs ~92% with it (target word is always known in advance here, since the app just
    spoke it via TTS — this isn't guessing, it's using information we already have).

    temperature is pinned to 0.0 deliberately: faster-whisper's default temperature is a
    fallback ladder (retries at increasing temperatures, i.e. randomness, when it doesn't like
    its own first-pass result) which made results non-deterministic — same audio, same params,
    different transcripts across repeat calls (1 in 5 in testing). Pinning to a single float
    disables that fallback loop entirely, trading a small amount of potential accuracy recovery
    for full reproducibility, which matters more for a drill-scoring loop."""

    async def transcribe(self, audio_bytes: bytes, language: str = "en", target_word: str | None = None) -> ASRResult:
        return await asyncio.to_thread(self._transcribe_sync, audio_bytes, language, target_word)

    def _transcribe_sync(self, audio_bytes: bytes, language: str, target_word: str | None) -> ASRResult:
        model = _get_model()
        t0 = time.monotonic()

        segments, info = model.transcribe(
            io.BytesIO(audio_bytes),
            language=language,
            beam_size=1,
            word_timestamps=True,
            initial_prompt=target_word,
            temperature=0.0,
        )
        segments = list(segments)

        transcript = " ".join(s.text.strip() for s in segments).strip()
        words = [
            {"word": w.word, "start": w.start, "end": w.end}
            for s in segments
            for w in (s.words or [])
        ]

        return ASRResult(
            transcript=transcript,
            language=info.language,
            latency_sec=round(time.monotonic() - t0, 3),
            words=words or None,
        )
