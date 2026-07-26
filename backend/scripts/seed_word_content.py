"""Seed the word_content table for English (Phase 1). Idempotent — safe to re-run.

Usage (from backend/, venv active, .env populated with a real DATABASE_URL):
    python -m scripts.seed_word_content
"""
import asyncio

from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.db.session import AsyncSessionLocal
from app.models.word_content import WordContent
from app.utils.ipa import to_ipa

LANGUAGE = "en"

WORDS_BY_LEVEL: dict[int, list[str]] = {
    1: [  # single syllable
        "cat", "cup", "sun", "dog", "pig", "hat", "bed", "red", "big", "run",
        "top", "ball", "book", "milk", "fish", "duck", "bus", "car", "star", "moon",
        "tree", "rain", "snow", "shoe", "house", "mouse", "cheese", "juice", "chair", "spoon",
        "plate", "brush", "frog", "bird", "girl", "boy", "man", "van", "fan", "pan",
        "box", "fox", "sock", "lock", "rock", "cake", "lake", "bike", "kite", "gate",
    ],
    2: [  # two syllables
        "rabbit", "butter", "monkey", "candy", "apple", "table", "pencil", "window", "garden", "basket",
        "cookie", "blanket", "jacket", "ladder", "mirror", "wagon", "tiger", "spider", "pumpkin", "penguin",
        "dolphin", "dragon", "rocket", "robot", "planet", "magnet", "pocket", "ticket", "zipper", "hammer",
        "pillow", "yellow", "turtle", "bottle", "puzzle", "muffin", "chicken", "kitten", "sister", "brother",
        "mommy", "daddy", "baby", "happy", "sunny", "funny", "water", "paper", "flower", "castle",
    ],
    3: [  # three syllables
        "elephant", "banana", "umbrella", "dinosaur", "butterfly", "tomato", "potato", "hospital", "telephone", "camera",
        "computer", "family", "animal", "chocolate", "octopus", "gorilla", "tornado", "volcano", "mosquito", "hamburger",
        "bicycle", "vitamin", "medicine", "grandmother", "grandfather", "yesterday", "tomorrow", "cucumber", "calendar", "cinnamon",
        "piano", "spaghetti", "kangaroo", "radio", "video", "studio", "stadium", "library", "alphabet", "astronaut",
        "continue", "remember", "celebrate", "orchestra", "broccoli", "pineapple", "popsicle", "adventure", "basketball", "helicopter",
    ],
    4: [  # short phrases
        "big red ball", "little brown dog", "happy blue bird", "funny green frog", "soft white cloud",
        "tall green tree", "small black cat", "warm sunny day", "cold rainy night", "sweet yellow banana",
        "fast red car", "slow green turtle", "pretty pink flower", "loud noisy truck", "quiet little mouse",
        "three little pigs", "five yellow ducks", "two happy kids", "my favorite toy", "our new house",
        "the big playground", "a tall tower", "her red shoes", "his blue hat", "our fluffy dog",
        "a shiny star", "the bright moon", "some cold water", "a warm blanket", "my best friend",
        "the funny clown", "a big surprise", "our little garden", "the tall giraffe", "a busy bee",
        "the sleepy cat", "my new book", "a round table", "the open door", "her long hair",
        "his fast bike", "a purple grape", "the soft pillow", "our happy family", "a scary monster",
        "the silly monkey", "my little sister", "a bright rainbow", "the deep ocean", "a wild animal",
    ],
    5: [  # one sentence max
        "I see a cat.", "I like my dog.", "She has a red ball.", "He can run fast.", "We play in the park.",
        "The sun is bright.", "My mom made cookies.", "I want some juice.", "The dog runs fast.", "She likes pink flowers.",
        "He has a blue car.", "We saw a big bird.", "I can jump high.", "The cat sleeps all day.", "My dad reads a book.",
        "I brush my teeth.", "She sings a happy song.", "He rides his bike.", "We eat lunch together.", "The baby is sleeping.",
        "I found a shiny rock.", "My sister likes ice cream.", "He wears a red hat.", "The frog jumps in water.", "I draw a big sun.",
        "She feeds the little fish.", "We walk to school.", "The bird flies away.", "I clean my room.", "He kicks the soccer ball.",
        "My friend has a puppy.", "The rain falls softly.", "I hear the loud thunder.", "She plants a small tree.", "We watch the bright stars.",
        "The turtle walks slowly.", "I open the front door.", "He climbs the tall tree.", "My grandma bakes a pie.", "The wind blows the leaves.",
        "I wash my hands.", "She paints a pretty picture.", "We build a sand castle.", "The moon shines at night.", "I catch a green frog.",
        "He throws the red ball.", "My teacher reads a story.", "The duck swims in the pond.", "I pet the soft rabbit.", "We fly a colorful kite.",
    ],
}


async def seed() -> None:
    total = 0
    async with AsyncSessionLocal() as db:
        for level, words in WORDS_BY_LEVEL.items():
            for word in words:
                ipa = to_ipa(word, language=LANGUAGE)
                stmt = (
                    pg_insert(WordContent)
                    .values(language=LANGUAGE, level=level, word=word, ipa=ipa)
                    .on_conflict_do_nothing(index_elements=["language", "level", "word"])
                )
                await db.execute(stmt)
                total += 1
            print(f"level {level}: {len(words)} words seeded ({sum(len(v) for k, v in WORDS_BY_LEVEL.items() if k <= level)}/250 so far)")
        await db.commit()
    print(f"done — {total} words processed for language={LANGUAGE}")


if __name__ == "__main__":
    asyncio.run(seed())
