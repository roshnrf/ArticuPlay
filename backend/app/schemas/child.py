import uuid
from pydantic import BaseModel, Field


class ChildCreate(BaseModel):
    name: str
    age: int = Field(ge=3, le=12)
    language: str = "en"
    current_level: int = Field(default=1, ge=1, le=5)
    avatar_url: str | None = None


class ChildUpdate(BaseModel):
    name: str | None = None
    age: int | None = Field(default=None, ge=3, le=12)
    language: str | None = None
    current_level: int | None = Field(default=None, ge=1, le=5)
    avatar_url: str | None = None


class ChildRead(BaseModel):
    id: uuid.UUID
    parent_id: uuid.UUID
    name: str
    age: int
    language: str
    current_level: int
    avatar_url: str | None
    total_sessions: int
    streak_days: int
    badges: str  # JSON string

    model_config = {"from_attributes": True}
