from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.word_content import WordContent
from app.schemas.word_content import WordContentRead


class WordContentService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def list_by_level(self, level: int, language: str) -> list[WordContentRead]:
        result = await self.db.scalars(
            select(WordContent)
            .where(WordContent.level == level, WordContent.language == language)
            .order_by(WordContent.word)
        )
        return [WordContentRead.model_validate(w) for w in result.all()]
