from fastapi import APIRouter, UploadFile, File, Form

from app.schemas.asr import ASRResult
from app.services.asr_service import ASRService
from app.services.phone_classifier_service import PhoneClassifierService
from app.utils.ipa import needs_phone_check

router = APIRouter()


@router.post("/transcribe", response_model=ASRResult)
async def transcribe(
    audio: UploadFile = File(..., description="WAV or WebM audio file"),
    language: str = Form("en"),
    target_word: str | None = Form(None, description="Unused by ASR decoding (see ASRService docstring — target-word biasing was removed after it masked real pronunciation errors). Used here to decide whether the phone classifier applies, then passed through to /session/score via compare_ipa."),
):
    """Transcribe child speech via self-hosted Whisper (faster-whisper), plus an audio-based
    phone-classifier check for words in its trained distribution (velar/rhotic, English) —
    see PhoneClassifierService for why transcript-only scoring isn't enough on its own."""
    audio_bytes = await audio.read()
    result = await ASRService().transcribe(audio_bytes, language=language)

    if target_word and needs_phone_check(target_word, language):
        result.phone_classifier_flag = await PhoneClassifierService().is_flagged_incorrect(audio_bytes)

    return result
