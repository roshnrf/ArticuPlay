import uuid
from pydantic import BaseModel


class WordContentRead(BaseModel):
    id: uuid.UUID
    language: str
    level: int
    word: str
    ipa: str
    image_url: str | None
    audio_url: str | None

    model_config = {"from_attributes": True}
