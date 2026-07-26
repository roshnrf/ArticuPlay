from fastapi import APIRouter
from fastapi.responses import Response

from app.schemas.tts import TTSRequest
from app.services.tts_service import TTSService

router = APIRouter()


@router.post("/tts")
async def synthesize(body: TTSRequest):
    """Speak a drill word via edge-tts and return the audio bytes."""
    audio_bytes = await TTSService().synthesize(body.word, language=body.language)
    return Response(content=audio_bytes, media_type="audio/mpeg")
