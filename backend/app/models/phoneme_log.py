import uuid
from datetime import date as date_type

from sqlalchemy import String, Date, Integer, ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class PhonemeLog(Base):
    """Daily per-phoneme accuracy tally for a child — powers the progress graph."""

    __tablename__ = "phoneme_logs"
    __table_args__ = (UniqueConstraint("child_id", "phoneme", "language", "date", name="uq_phoneme_log_day"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    child_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("children.id"), nullable=False, index=True)
    phoneme: Mapped[str] = mapped_column(String(10), nullable=False)  # single IPA symbol
    language: Mapped[str] = mapped_column(String(5), nullable=False)
    date: Mapped[date_type] = mapped_column(Date, nullable=False)
    correct: Mapped[int] = mapped_column(Integer, default=0)
    incorrect: Mapped[int] = mapped_column(Integer, default=0)
