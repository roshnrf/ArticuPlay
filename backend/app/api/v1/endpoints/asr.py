from fastapi import APIRouter, UploadFile, File, Form

from app.schemas.asr import ASRResult
from app.services.asr_service import ASRService

router = APIRouter()


@router.post("/transcribe", response_model=ASRResult)
async def transcribe(
    audio: UploadFile = File(..., description="WAV or WebM audio file"),
    language: str = Form("en"),
    target_word: str | None = Form(None, description="Unused by ASR decoding (see ASRService docstring — target-word biasing was removed after it masked real pronunciation errors); still scored against separately by /session/score via compare_ipa."),
):
    """Transcribe child speech via self-hosted Whisper (faster-whisper)."""
    audio_bytes = await audio.read()
    return await ASRService().transcribe(audio_bytes, language=language)
