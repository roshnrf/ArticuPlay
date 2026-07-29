import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, String, DateTime, Integer, Float, Boolean, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class DrillItem(Base):
    """One target word/phrase attempt within a drill session."""

    __tablename__ = "drill_items"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("drill_sessions.id"), nullable=False, index=True)
    language: Mapped[str] = mapped_column(String(5), nullable=False)
    item_index: Mapped[int] = mapped_column(Integer, nullable=False)  # position within the session
    target_word: Mapped[str] = mapped_column(String(200), nullable=False)
    target_ipa: Mapped[str] = mapped_column(String(200), nullable=False)
    child_audio_url: Mapped[str] = mapped_column(String, nullable=True)  # Supabase Storage path
    child_transcript: Mapped[str] = mapped_column(String(200), nullable=True)
    child_ipa: Mapped[str] = mapped_column(String(200), nullable=True)
    accuracy_score: Mapped[float] = mapped_column(Float, nullable=True)  # 0-1
    # list of {"position": int, "expected": str|None, "got": str|None, "type": str} dicts —
    # a single attempt can have multiple phoneme errors, so this can't be a single column.
    # type is one of: substitution | omission | addition | cluster_reduction | syllable_deletion
    errors: Mapped[list] = mapped_column(JSON, default=list)
    attempt_num: Mapped[int] = mapped_column(Integer, default=1)  # 1-3, auto-advance after 3
    passed: Mapped[bool] = mapped_column(Boolean, default=False)
    # True when compare_ipa scored this as a pass but the phone classifier (trained on real
    # disordered speech) disagreed and downgraded it — see session_service.score(). Tracked
    # separately from `errors` since it's a different signal (audio-based, not transcript-based).
    phone_classifier_override: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    session: Mapped["DrillSession"] = relationship("DrillSession", back_populates="items")
