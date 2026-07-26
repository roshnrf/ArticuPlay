from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


# Import all models here so Alembic can detect them
from app.models import (  # noqa: F401, E402
    user,
    child,
    drill_session,
    drill_item,
    phoneme_log,
    word_content,
    parent_tip,
)
