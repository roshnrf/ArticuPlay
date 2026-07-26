"""Adds 50 new English words to word_content (10 per level) — real content
growth, not just a test artifact, and doubles as a genuine held-out
generalization test since the fine-tuned model has never seen these words.
Idempotent (ON CONFLICT DO NOTHING), safe to re-run.
"""
import asyncio

from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.db.session import AsyncSessionLocal
from app.models.word_content import WordContent
from app.utils.ipa import to_ipa

LANGUAGE = "en"

NEW_WORDS_BY_LEVEL: dict[int, list[str]] = {
    1: ["mug", "hen", "pot", "jar", "key", "doll", "ring", "sled", "crib", "bell"],
    2: ["carrot", "hairbrush", "lizard", "bunny", "pretzel", "whistle", "ribbon", "needle", "purple", "garlic"],
    3: ["xylophone", "microphone", "ambulance", "caravan", "symphony", "energy", "interest", "beautiful", "important", "newspaper"],
    4: [
        "little yellow duck", "big blue whale", "soft pink pillow", "three red apples", "the happy puppy",
        "her tall brother", "his new shoes", "a cold drink", "the warm sun", "our favorite game",
    ],
    5: [
        "I love my mom.", "She has a big smile.", "He plays with blocks.", "We go to the park.",
        "The girl has a doll.", "My cat likes milk.", "I can read a book.", "She draws a flower.",
        "He jumps very high.", "We sing a song together.",
    ],
}


async def seed() -> None:
    total = 0
    async with AsyncSessionLocal() as db:
        for level, words in NEW_WORDS_BY_LEVEL.items():
            for word in words:
                ipa = to_ipa(word, language=LANGUAGE)
                stmt = (
                    pg_insert(WordContent)
                    .values(language=LANGUAGE, level=level, word=word, ipa=ipa)
                    .on_conflict_do_nothing(index_elements=["language", "level", "word"])
                )
                await db.execute(stmt)
                total += 1
            print(f"level {level}: {len(words)} new words seeded")
        await db.commit()
    print(f"done — {total} new words processed")


if __name__ == "__main__":
    asyncio.run(seed())
