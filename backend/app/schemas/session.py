import uuid
from datetime import datetime
from pydantic import BaseModel


class SessionCreate(BaseModel):
    child_id: uuid.UUID
    language: str
    level: int


class SessionRead(BaseModel):
    id: uuid.UUID
    child_id: uuid.UUID
    language: str
    level: int
    status: str
    items_count: int
    correct_count: int
    started_at: datetime
    completed_at: datetime | None

    model_config = {"from_attributes": True}


class ScoreRequest(BaseModel):
    session_id: uuid.UUID
    item_index: int
    target_word: str
    language: str = "en"
    child_transcript: str
    attempt_num: int = 1


class PhonemeErrorRead(BaseModel):
    position: int
    expected: str | None
    got: str | None
    type: str


class ScoreResult(BaseModel):
    item_index: int
    accuracy_score: float
    errors: list[PhonemeErrorRead]
    passed: bool
    attempt_num: int
