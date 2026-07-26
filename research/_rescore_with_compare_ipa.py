"""Re-scores the last full 250-word test results using the REAL product
scoring method (compare_ipa, phoneme-level) instead of exact-substring match,
to see the true accuracy picture the product would actually show."""
import json
import sys

sys.path.insert(0, "/mnt/c/Users/rosha/Documents/sw_2/backend")
from app.utils.ipa import to_ipa
from app.utils.compare_ipa import compare_ipa

PASS_THRESHOLD = 0.8

with open("/tmp/full_prodword_results.json") as f:
    results = json.load(f)

base_correct = 0
lora_correct = 0
per_level = {}

for r in results:
    target_ipa = to_ipa(r["word"], language="en")
    base_ipa = to_ipa(r["base_text"], language="en")
    lora_ipa = to_ipa(r["lora_text"], language="en")

    base_score = compare_ipa(target_ipa, base_ipa).accuracy
    lora_score = compare_ipa(target_ipa, lora_ipa).accuracy

    base_ok = base_score >= PASS_THRESHOLD
    lora_ok = lora_score >= PASS_THRESHOLD
    base_correct += base_ok
    lora_correct += lora_ok

    lvl = r["level"]
    per_level.setdefault(lvl, {"base": 0, "lora": 0, "n": 0})
    per_level[lvl]["base"] += base_ok
    per_level[lvl]["lora"] += lora_ok
    per_level[lvl]["n"] += 1

n = len(results)
print(f"=== Re-scored with compare_ipa (phoneme-level, threshold={PASS_THRESHOLD}) ===")
print(f"base: {base_correct}/{n} ({base_correct/n*100:.1f}%)   lora: {lora_correct}/{n} ({lora_correct/n*100:.1f}%)")
print("\nper level:")
for lvl in sorted(per_level):
    d = per_level[lvl]
    print(f"  level {lvl}: base {d['base']}/{d['n']} ({d['base']/d['n']*100:.0f}%)   lora {d['lora']}/{d['n']} ({d['lora']/d['n']*100:.0f}%)")
