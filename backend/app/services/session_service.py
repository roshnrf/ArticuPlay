from dataclasses import asdict
from datetime import datetime, timezone
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.child import Child
from app.models.drill_item import DrillItem
from app.models.drill_session import DrillSession
from app.schemas.session import ScoreRequest, ScoreResult, SessionCreate, SessionRead
from app.services.phoneme_log_service import PhonemeLogService
from app.utils.compare_ipa import compare_ipa
from app.utils.ipa import to_ipa

PASS_THRESHOLD = 0.8


class SessionService:
    """Drill session lifecycle: start, item-level scoring (compare_ipa), and completion.
    Every operation is scoped to the requesting parent — a session/child is never visible
    or writable by anyone else, same discipline as ChildService."""

    def __init__(self, db: AsyncSession, parent_id: UUID):
        self.db = db
        self.parent_id = parent_id

    async def _get_owned_child(self, child_id: UUID) -> Child:
        child = await self.db.get(Child, child_id)
        if not child or child.parent_id != self.parent_id:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Child not found")
        return child

    async def _get_owned_session(self, session_id: UUID) -> DrillSession:
        session = await self.db.get(DrillSession, session_id)
        if not session:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")
        await self._get_owned_child(session.child_id)  # raises 404 if not owned
        return session

    async def start(self, body: SessionCreate) -> SessionRead:
        await self._get_owned_child(body.child_id)
        session = DrillSession(child_id=body.child_id, language=body.language, level=body.level)
        self.db.add(session)
        await self.db.commit()
        await self.db.refresh(session)
        return SessionRead.model_validate(session)

    async def get(self, session_id: UUID) -> SessionRead:
        session = await self._get_owned_session(session_id)
        return SessionRead.model_validate(session)

    async def list_for_child(self, child_id: UUID) -> list[SessionRead]:
        await self._get_owned_child(child_id)
        result = await self.db.scalars(select(DrillSession).where(DrillSession.child_id == child_id))
        return [SessionRead.model_validate(s) for s in result.all()]

    async def score(self, body: ScoreRequest) -> ScoreResult:
        session = await self._get_owned_session(body.session_id)

        target_ipa = to_ipa(body.target_word, language=body.language)
        child_ipa = to_ipa(body.child_transcript, language=body.language)
        result = compare_ipa(target_ipa, child_ipa)
        passed = result.accuracy >= PASS_THRESHOLD
        errors_as_dicts = [asdict(e) for e in result.errors]

        # One-directional gate: the phone classifier only ever downgrades a transcript-based
        # pass, never upgrades a fail. compare_ipa scores the ASR transcript, which can't catch
        # a real articulation error when the transcript still reads as the right word (measured
        # false-accept 15-28% on real disordered speech — see tasks/lessons.md). The classifier
        # judges the raw audio directly, so it catches some of what the transcript misses. It's
        # not applied the other way (fail->pass) because its own error rate could just trade one
        # failure mode for another; only closing the clinically dangerous direction is worth the risk.
        phone_classifier_override = passed and body.phone_classifier_flag is True
        if phone_classifier_override:
            passed = False

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
            phone_classifier_override=phone_classifier_override,
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
            phone_classifier_override=phone_classifier_override,
        )

    async def complete(self, session_id: UUID) -> SessionRead:
        session = await self._get_owned_session(session_id)

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
