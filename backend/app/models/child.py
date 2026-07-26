import uuid
from datetime import datetime, timezone

from sqlalchemy import String, DateTime, Integer, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Child(Base):
    """Child profile linked to a parent account."""

    __tablename__ = "children"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    parent_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    age: Mapped[int] = mapped_column(Integer, nullable=False)
    language: Mapped[str] = mapped_column(String(5), default="en")  # en | ar | hi | zh
    current_level: Mapped[int] = mapped_column(Integer, default=1)  # 1–5
    avatar_url: Mapped[str] = mapped_column(String, nullable=True)
    total_sessions: Mapped[int] = mapped_column(Integer, default=0)
    streak_days: Mapped[int] = mapped_column(Integer, default=0)
    badges: Mapped[str] = mapped_column(String, default="[]")  # JSON list of badge IDs
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    parent: Mapped["User"] = relationship("User", back_populates="children")
    sessions: Mapped[list["DrillSession"]] = relationship(
        "DrillSession", back_populates="child", cascade="all, delete-orphan"
    )
