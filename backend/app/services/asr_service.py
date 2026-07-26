import asyncio
import io
import time
from functools import lru_cache
from pathlib import Path

from faster_whisper import WhisperModel

from app.core.config import settings
from app.schemas.asr import ASRResult

BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent


@lru_cache(maxsize=1)
def _get_model() -> WhisperModel:
    """Load the Whisper model once per process. Loading takes ~1-2s; must never happen per-request."""
    if settings.WHISPER_MODEL_PATH:
        model_path = BACKEND_ROOT / settings.WHISPER_MODEL_PATH
        return WhisperModel(str(model_path), device="cpu", compute_type="int8")
    return WhisperModel(settings.WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")


class ASRService:
    """Self-hosted speech-to-text via faster-whisper, running our LoRA fine-tuned Whisper-small
    (see research/uxtd_transcriber — 96% phoneme accuracy on held-out vocabulary vs 80% for stock
    Whisper, verified across 3 training seeds).

    initial_prompt is deliberately NOT set to target_word (changed 2026-07-26). Ablation testing
    on real disordered child speech (UXSSD corpus) found target-word biasing pushed false-accept
    to 28% — the model would transcribe back the correct spelling even when the child's actual
    phone was wrong (e.g. 'tiger' with a genuinely bad /g/ still decoded as 'tiger'), because the
    bias overrides subtle acoustic error signal. Removing it drops false-accept to 15% at the cost
    of raising false-reject 32%->44% — a real tradeoff, not a free win, but the false-accept
    direction is the clinically dangerous one (reinforces a wrong articulation pattern), so it's
    the safer default. 15% false-accept still remains — full fix needs routing through the
    phone-level classifier (research/phone_classifier) instead of ASR-transcript scoring; not
    yet integrated. See tasks/lessons.md for the full ablation numbers.

    beam_size=5 (up from 1) matches the config actually validated in eval — beam search fixed
    most premature-truncation/repetition-loop failures at zero retraining cost.

    temperature is pinned to 0.0 deliberately: faster-whisper's default temperature is a
    fallback ladder (retries at increasing temperatures, i.e. randomness, when it doesn't like
    its own first-pass result) which made results non-deterministic — same audio, same params,
    different transcripts across repeat calls (1 in 5 in testing). Pinning to a single float
    disables that fallback loop entirely, trading a small amount of potential accuracy recovery
    for full reproducibility, which matters more for a drill-scoring loop."""

    async def transcribe(self, audio_bytes: bytes, language: str = "en") -> ASRResult:
        return await asyncio.to_thread(self._transcribe_sync, audio_bytes, language)

    def _transcribe_sync(self, audio_bytes: bytes, language: str) -> ASRResult:
        model = _get_model()
        t0 = time.monotonic()

        segments, info = model.transcribe(
            io.BytesIO(audio_bytes),
            language=language,
            beam_size=5,
            word_timestamps=True,
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
