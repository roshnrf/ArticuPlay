import uuid

from sqlalchemy import String, Integer, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class WordContent(Base):
    """Drill word/phrase bank, seeded per language+level. IPA precomputed at seed time."""

    __tablename__ = "word_content"
    __table_args__ = (UniqueConstraint("language", "level", "word", name="uq_word_content_entry"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    language: Mapped[str] = mapped_column(String(5), nullable=False, index=True)
    level: Mapped[int] = mapped_column(Integer, nullable=False)  # 1-5
    word: Mapped[str] = mapped_column(String(200), nullable=False)
    ipa: Mapped[str] = mapped_column(String(200), nullable=False)
    image_url: Mapped[str] = mapped_column(String, nullable=True)
    audio_url: Mapped[str] = mapped_column(String, nullable=True)  # optional pre-baked cache; TTS is generated live
