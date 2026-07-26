"""Generates synthetic audio for the v2 (50) + v3 (191) word-bank expansion —
folds them into the trainable vocabulary now that the generalization test on
v2's words already ran. Writes data/augment_v2v3/ + data/augment_v2v3.csv."""
import csv
import json
import subprocess
import urllib.request
from pathlib import Path

BACKEND = "http://127.0.0.1:8000/api/v1"
DATA_DIR = Path(__file__).parent / "data"
AUGMENT_DIR = DATA_DIR / "augment_v2v3"

V2_WORDS = {
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

V3_WORDS = {
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


def synthesize(text: str, mp3_path: Path) -> None:
    body = json.dumps({"word": text, "language": "en"}).encode()
    req = urllib.request.Request(
        f"{BACKEND}/session/tts", data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        mp3_path.write_bytes(resp.read())


def main() -> None:
    AUGMENT_DIR.mkdir(parents=True, exist_ok=True)
    all_words = [w for ws in V2_WORDS.values() for w in ws] + [w for ws in V3_WORDS.values() for w in ws]
    print(f"generating synthetic audio for {len(all_words)} words (v2 held-out + v3 expansion)...")

    manifest_rows = []
    for i, word in enumerate(all_words):
        mp3_path = AUGMENT_DIR / f"aug_{i}.mp3"
        wav_path = AUGMENT_DIR / f"aug_{i}.wav"
        synthesize(word, mp3_path)
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(mp3_path), "-ar", "16000", "-ac", "1", str(wav_path), "-loglevel", "error"],
            check=True,
        )
        manifest_rows.append({"wav_path": str(wav_path.relative_to(DATA_DIR)), "text": word, "speaker": "SYNTH"})
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(all_words)}")

    with open(DATA_DIR / "augment_v2v3.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["wav_path", "text", "speaker"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"done — {len(manifest_rows)} examples saved to data/augment_v2v3.csv")


if __name__ == "__main__":
    main()
