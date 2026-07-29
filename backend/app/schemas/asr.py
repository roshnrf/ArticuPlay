from pydantic import BaseModel


class ASRResult(BaseModel):
    transcript: str
    language: str
    latency_sec: float
    words: list[dict] | None = None  # word-level timestamps from faster-whisper
    phone_classifier_flag: bool | None = None  # True = phone classifier thinks this attempt is wrong;
    # None = not applicable (word outside the classifier's trained velar/rhotic distribution)
