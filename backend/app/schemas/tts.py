from pydantic import BaseModel


class TTSRequest(BaseModel):
    word: str
    language: str = "en"  # en | ar | hi | zh
