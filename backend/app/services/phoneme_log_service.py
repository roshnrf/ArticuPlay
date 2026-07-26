from datetime import timezone, datetime
from uuid import UUID

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.phoneme_log import PhonemeLog
from app.utils.compare_ipa import PhonemeError

_ERROR_TYPES_COUNTING_AS_INCORRECT = {"substitution", "omission", "cluster_reduction", "syllable_deletion"}


class PhonemeLogService:
    """Tallies per-phoneme correct/incorrect counts for the day, one row per
    (child, phoneme, language, date) — upserted so repeat attempts accumulate."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def record_attempt(
        self,
        child_id: UUID,
        language: str,
        target_phonemes: list[str],
        errors: list[PhonemeError],
    ) -> None:
        incorrect_positions = {e.position for e in errors if e.type in _ERROR_TYPES_COUNTING_AS_INCORRECT}
        today = datetime.now(timezone.utc).date()

        tallies: dict[str, dict[str, int]] = {}
        for position, phoneme in enumerate(target_phonemes):
            bucket = tallies.setdefault(phoneme, {"correct": 0, "incorrect": 0})
            if position in incorrect_positions:
                bucket["incorrect"] += 1
            else:
                bucket["correct"] += 1

        for phoneme, counts in tallies.items():
            stmt = pg_insert(PhonemeLog).values(
                child_id=child_id,
                phoneme=phoneme,
                language=language,
                date=today,
                correct=counts["correct"],
                incorrect=counts["incorrect"],
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=["child_id", "phoneme", "language", "date"],
                set_={
                    "correct": PhonemeLog.correct + stmt.excluded.correct,
                    "incorrect": PhonemeLog.incorrect + stmt.excluded.incorrect,
                },
            )
            await self.db.execute(stmt)
