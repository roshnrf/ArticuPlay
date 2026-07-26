from dataclasses import asdict
from datetime import datetime, timezone
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.drill_item import DrillItem
from app.models.drill_session import DrillSession
from app.schemas.session import ScoreRequest, ScoreResult, SessionCreate, SessionRead
from app.services.phoneme_log_service import PhonemeLogService
from app.utils.compare_ipa import compare_ipa
from app.utils.ipa import to_ipa

PASS_THRESHOLD = 0.8


class SessionService:
    """Drill session lifecycle: start, item-level scoring (compare_ipa), and completion."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def start(self, body: SessionCreate) -> SessionRead:
        session = DrillSession(child_id=body.child_id, language=body.language, level=body.level)
        self.db.add(session)
        await self.db.commit()
        await self.db.refresh(session)
        return SessionRead.model_validate(session)

    async def get(self, session_id: UUID) -> SessionRead:
        session = await self.db.get(DrillSession, session_id)
        if not session:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")
        return SessionRead.model_validate(session)

    async def list_for_child(self, child_id: UUID) -> list[SessionRead]:
        result = await self.db.scalars(select(DrillSession).where(DrillSession.child_id == child_id))
        return [SessionRead.model_validate(s) for s in result.all()]

    async def score(self, body: ScoreRequest) -> ScoreResult:
        session = await self.db.get(DrillSession, body.session_id)
        if not session:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")

        target_ipa = to_ipa(body.target_word, language=body.language)
        child_ipa = to_ipa(body.child_transcript, language=body.language)
        result = compare_ipa(target_ipa, child_ipa)
        passed = result.accuracy >= PASS_THRESHOLD
        errors_as_dicts = [asdict(e) for e in result.errors]

        item = DrillItem(
            session_id=body.session_id,
            language=body.language,
            item_index=body.item_index,
            target_word=body.target_word,
            target_ipa=target_ipa,
            child_transcript=body.child_transcript,
            child_ipa=child_ipa,
            accuracy_score=result.accuracy,
            errors=errors_as_dicts,
            attempt_num=body.attempt_num,
            passed=passed,
        )
        self.db.add(item)

        await PhonemeLogService(self.db).record_attempt(
            child_id=session.child_id,
            language=body.language,
            target_phonemes=result.target_phonemes,
            errors=result.errors,
        )

        await self.db.commit()

        return ScoreResult(
            item_index=body.item_index,
            accuracy_score=result.accuracy,
            errors=errors_as_dicts,
            passed=passed,
            attempt_num=body.attempt_num,
        )

    async def complete(self, session_id: UUID) -> SessionRead:
        session = await self.db.get(DrillSession, session_id)
        if not session:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")

        passed_flags = (
            await self.db.scalars(select(DrillItem.passed).where(DrillItem.session_id == session_id))
        ).all()

        session.items_count = len(passed_flags)
        session.correct_count = sum(passed_flags)
        session.status = "completed"
        session.completed_at = datetime.now(timezone.utc)
        await self.db.commit()
        await self.db.refresh(session)
        return SessionRead.model_validate(session)
