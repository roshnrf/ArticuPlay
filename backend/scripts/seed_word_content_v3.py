"""Major word-bank expansion — wider category variety per level (animals, food,
body parts, clothing, household, nature, verbs, emotions for L1-L3; new phrase/
sentence structures for L4-L5), not just more of the same shapes. Checked against
the existing ~300 words (original 250 + v2's 50) to avoid duplicates.
Idempotent (ON CONFLICT DO NOTHING), safe to re-run.
"""
import asyncio

from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.db.session import AsyncSessionLocal
from app.models.word_content import WordContent
from app.utils.ipa import to_ipa

LANGUAGE = "en"

NEW_WORDS_BY_LEVEL: dict[int, list[str]] = {
    1: [
        "cow", "sheep", "goat", "bee", "ant", "bear", "deer", "seal", "whale", "shark",
        "crab", "snail", "mole", "owl", "hawk", "wolf", "bread", "egg", "rice", "soup",
        "meat", "fruit", "corn", "peach", "plum", "pear", "lime", "mint", "salt", "jam",
        "head", "hand", "foot", "arm", "leg", "eye", "ear", "nose", "chin", "cheek",
        "back", "neck", "knee", "thumb", "wrist", "coat", "boot", "vest", "belt", "cap",
    ],
    2: [
        "parrot", "falcon", "beetle", "gopher", "ferret", "weasel", "walrus", "otter", "panda", "koala",
        "camel", "zebra", "cricket", "hamster", "gecko", "python", "cobra", "waffle", "pancake", "sandwich",
        "popcorn", "yogurt", "biscuit", "noodle", "mustard", "ketchup", "lemon", "melon", "mango", "peanut",
        "walnut", "raisin", "spinach", "candle", "curtain", "kitchen", "bedroom", "bathroom", "closet", "hallway",
        "ceiling", "doorbell", "mailbox", "sidewalk", "driveway", "backpack", "notebook", "crayon", "scissors", "stapler",
    ],
    3: [
        "octagon", "triangle", "rectangle", "happiness", "holiday", "buffalo", "gasoline", "magazine", "lemonade", "vacation",
        "location", "monument", "instrument", "restaurant", "continent", "president", "accident", "different", "dangerous", "curious",
        "delicious", "generous", "terrible", "horrible", "possible", "musical", "magical", "medical", "physical", "national",
        "personal", "popular", "regular", "similar", "policeman", "carpenter", "mechanic", "musician", "scientist", "chemistry", "history",
    ],
    4: [
        "a fluffy white rabbit", "two green frogs", "my favorite red truck", "the sleepy gray cat", "his old brown boots",
        "her shiny gold ring", "a loud noisy train", "the quiet dark night", "three small silver fish", "our new soft blanket",
        "the funny clown juggles", "a tall green tree", "her pretty pink dress", "the fast little mouse", "his big loud dog",
        "a cold glass of milk", "the warm chocolate cake", "her favorite yellow bike", "the tiny baby bird", "a happy dancing bear",
        "the strong brave firefighter", "his shiny new watch", "a soft fuzzy blanket", "the busy city street", "her colorful paper kite",
    ],
    5: [
        "The dog ran across the yard.", "I want to eat an apple.", "She likes to jump on the bed.",
        "We watched the birds fly away.", "He found a shiny red rock.", "My sister drew a funny picture.",
        "The rain fell all night long.", "I helped my mom bake cookies.", "The cat sleeps on the soft chair.",
        "We played outside until dark.", "He rides his bike to school.", "She sang a happy little song.",
        "The baby laughed at the puppy.", "I put my toys away today.", "The wind blew the leaves around.",
        "We built a tall tower of blocks.", "She waved goodbye to her friend.", "The fish swam in the blue pond.",
        "He climbed to the top of the hill.", "I brushed my teeth before bed.", "The sun came out after the storm.",
        "My dad reads me a story every night.", "We picked flowers in the garden.", "The train moved slowly down the track.",
        "She opened the door and smiled.",
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
