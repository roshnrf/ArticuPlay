import json
import jiwer

with open("/tmp/full_prodword_results.json") as f:
    results = json.load(f)

refs = [r["word"] for r in results]
base_preds = [r["base_text"] for r in results]
lora_preds = [r["lora_text"] for r in results]

print(f"n = {len(results)}")
print(f"base WER  = {jiwer.wer(refs, base_preds):.4f}")
print(f"lora WER  = {jiwer.wer(refs, lora_preds):.4f}")
print(f"base CER  = {jiwer.cer(refs, base_preds):.4f}")
print(f"lora CER  = {jiwer.cer(refs, lora_preds):.4f}")

# per-level WER
per_level = {}
for r in results:
    per_level.setdefault(r["level"], {"refs": [], "base": [], "lora": []})
    per_level[r["level"]]["refs"].append(r["word"])
    per_level[r["level"]]["base"].append(r["base_text"])
    per_level[r["level"]]["lora"].append(r["lora_text"])

print("\nper-level WER:")
for lvl in sorted(per_level):
    d = per_level[lvl]
    bw = jiwer.wer(d["refs"], d["base"])
    lw = jiwer.wer(d["refs"], d["lora"])
    print(f"  level {lvl}: base WER={bw:.3f}  lora WER={lw:.3f}")
