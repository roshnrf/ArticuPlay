from uuid import UUID

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import get_current_user
from app.db.session import get_db
from app.models.user import User
from app.schemas.session import ScoreRequest, ScoreResult, SessionCreate, SessionRead
from app.services.session_service import SessionService

router = APIRouter()


def _service(db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)) -> SessionService:
    return SessionService(db, parent_id=user.id)


@router.post("/start", response_model=SessionRead, status_code=status.HTTP_201_CREATED)
async def start_session(body: SessionCreate, service: SessionService = Depends(_service)):
    """Start a new drill session for a child."""
    return await service.start(body)


@router.post("/score", response_model=ScoreResult)
async def score_item(body: ScoreRequest, service: SessionService = Depends(_service)):
    """Score one drill attempt: target vs child transcript, phoneme by phoneme."""
    return await service.score(body)


@router.post("/complete", response_model=SessionRead)
async def complete_session(session_id: UUID, service: SessionService = Depends(_service)):
    """Mark a session complete and roll up its item stats."""
    return await service.complete(session_id)


@router.get("/child/{child_id}", response_model=list[SessionRead])
async def list_sessions(child_id: UUID, service: SessionService = Depends(_service)):
    return await service.list_for_child(child_id)


@router.get("/{session_id}", response_model=SessionRead)
async def get_session(session_id: UUID, service: SessionService = Depends(_service)):
    return await service.get(session_id)
