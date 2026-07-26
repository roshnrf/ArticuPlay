from pydantic import BaseModel


class ASRResult(BaseModel):
    transcript: str
    language: str
    latency_sec: float
    words: list[dict] | None = None  # word-level timestamps from faster-whisper
