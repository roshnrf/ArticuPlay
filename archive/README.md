# Archive

Frozen snapshots of code from earlier stages of the project, kept so every
reported number stays traceable to the exact code that produced it — not
just the current, most-evolved version of each script.

Most stage-to-stage progress in this repo is just separate files
(`train.py` → `train_lora.py` → `train_lora_augmented.py` →
`train_lora_fullvocab.py`, each a distinct step) or ordinary git history.
The three files here exist because those two mechanisms weren't enough:
the underlying scripts were edited **in place** for a later stage, and
the edit happened *before this repo's first commit* — so neither "look at
an earlier file" nor "check git log" could recover the earlier version.
These are manual reconstructions of that pre-edit state, kept purely for
traceability. **Do not run these — they're not maintained and may
reference paths/models that no longer exist.**

| File | What it produced | Superseded by |
|---|---|---|
| `train_lora_fullvocab_250vocab.py` | The 250-word-vocab checkpoint | `research/uxtd_transcriber/train_lora_fullvocab.py` (edited in place to add `--seed` + the 491-word vocabulary expansion) |
| `evaluate_new_words_heldout_50word_v2.py` | "82.7% → 95.2%" (n=50, tested against the 250-vocab checkpoint) | `research/uxtd_transcriber/evaluate_new_words_heldout.py` (edited in place to a fresh 30-word set once the original 50 words got folded into training) |
| `asr_service_v1_base_biased.py` | Original production ASR: stock Whisper "base", target-word decoding bias, `beam_size=1` | `backend/app/services/asr_service.py` — later changes (fine-tuned model + unbiased decoding, then the phone-classifier gate) *are* in that file's git history, since those happened after this repo's first commit |

See the root [README.md](../README.md#model-version-history) for the full
stage-by-stage numbers.
