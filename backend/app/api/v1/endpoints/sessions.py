from uuid import UUID

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.session import ScoreRequest, ScoreResult, SessionCreate, SessionRead
from app.services.session_service import SessionService

router = APIRouter()


@router.post("/start", response_model=SessionRead, status_code=status.HTTP_201_CREATED)
async def start_session(body: SessionCreate, db: AsyncSession = Depends(get_db)):
    """Start a new drill session for a child."""
    return await SessionService(db).start(body)


@router.post("/score", response_model=ScoreResult)
async def score_item(body: ScoreRequest, db: AsyncSession = Depends(get_db)):
    """Score one drill attempt: target vs child transcript, phoneme by phoneme."""
    return await SessionService(db).score(body)


@router.post("/complete", response_model=SessionRead)
async def complete_session(session_id: UUID, db: AsyncSession = Depends(get_db)):
    """Mark a session complete and roll up its item stats."""
    return await SessionService(db).complete(session_id)


@router.get("/child/{child_id}", response_model=list[SessionRead])
async def list_sessions(child_id: UUID, db: AsyncSession = Depends(get_db)):
    return await SessionService(db).list_for_child(child_id)


@router.get("/{session_id}", response_model=SessionRead)
async def get_session(session_id: UUID, db: AsyncSession = Depends(get_db)):
    return await SessionService(db).get(session_id)
