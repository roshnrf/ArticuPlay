import uuid
from datetime import datetime, timezone

from sqlalchemy import String, DateTime, Integer, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class DrillSession(Base):
    """One drill session: a run of 8-10 word/phrase items at a given level+language."""

    __tablename__ = "drill_sessions"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    child_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("children.id"), nullable=False, index=True)
    language: Mapped[str] = mapped_column(String(5), nullable=False)
    level: Mapped[int] = mapped_column(Integer, nullable=False)  # 1-5
    status: Mapped[str] = mapped_column(String(20), default="in_progress")  # in_progress | completed
    items_count: Mapped[int] = mapped_column(Integer, default=0)
    correct_count: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)

    child: Mapped["Child"] = relationship("Child", back_populates="sessions")
    items: Mapped[list["DrillItem"]] = relationship(
        "DrillItem", back_populates="session", cascade="all, delete-orphan"
    )
