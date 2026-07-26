from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.word_content import WordContentRead
from app.services.word_content_service import WordContentService

router = APIRouter()


@router.get("/words/{level}", response_model=list[WordContentRead])
async def list_words(level: int, lang: str = Query("en", alias="lang"), db: AsyncSession = Depends(get_db)):
    return await WordContentService(db).list_by_level(level, lang)
