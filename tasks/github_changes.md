# GitHub changes to make (written 2026-11-26)

Repo: https://github.com/roshnrf/ArticuPlay — local `main` = `origin/main` (last commit 2026-07-29).
Status as of 2026-11-26: done on branch `docs/nov-rebuild-fixes` unless still unchecked.

## 1. Fix claims in README that are wrong or misleading

- [x] **README line 100–102**: says UXSSD is "real children with CAS" and that it was measured
      "through the live deployed pipeline". Both are wrong. UXSSD is children with
      *speech sound disorders* (not a confirmed CAS diagnosis), and the app has never been
      deployed. Change to: "children with speech sound disorders (UltraSuite UXSSD,
      therapist-scored), measured through the full local pipeline".
- [x] **README line 95–96 (and line 60, 82)**: the 96.0% vs 80.0% held-out test uses
      **synthetic TTS audio** (edge-tts), not recordings of children. Say so next to the number.
- [x] **README intro (line 10–12)**: add one plain sentence: "No child has used the app yet;
      all testing is on corpus recordings, synthetic audio, and the developer's own voice."
- [x] **README line 113**: "a few real-correct attempts get wrongly downgraded" undersells a
      46% false-reject rate. Reword: "about half of correct attempts are still rejected; this is
      the main open problem."

## 2. Add what's missing from README

- [x] **November rebuild result**: "Rebuilt from this repo in Nov 2026 after the original
      machine's data was lost: retrained model scores 96.7% vs 86.7% on the same 30 held-out
      words." (Stock Whisper differs because the TTS test audio is regenerated with a newer
      edge-tts.) Seeds 43/44 were not re-run.
- [x] **Setup / run instructions** (there are none). Cover: WSL2 Ubuntu, `espeak-ng`, `ffmpeg`,
      PostgreSQL, Python 3.12 venv + `backend/requirements.txt`, `.env` from `.env.example`,
      `alembic upgrade head`, the three `seed_word_content*.py` scripts, getting the models
      (see section 3), `uvicorn`, and the frontend (`npm install`, vite on :5173).
- [x] **How to get the transcriber model**: `backend/ml_models/` is gitignored, so a fresh clone
      cannot run the app without retraining. Document the retrain → merge → CTranslate2 steps
      (`train_lora_fullvocab.py`, `merge_lora_for_deploy.py`, ct2 conversion), or publish the
      weights (section 3).
- [ ] **Demo video or GIF** once recorded.

## 3. Decide: publish the model weights?

- [x] Decided 2026-11-26: not published for now; README documents how to retrain. The LoRA adapter is ~7 MB (`research/uxtd_transcriber/uxtd_whisper_small_lora_fullvocab/final`).
      Options: GitHub Release asset, or a Hugging Face model repo. The phone classifier weights
      are already in the repo.
- [x] Licence point (README states weights carry the non-commercial restriction): it was trained on UltraSuite (CC BY-NC 4.0). Shared weights should carry the
      same non-commercial note and a citation of UltraSuite.

## 4. Commit the November fixes (currently uncommitted on this PC)

Split into separate commits, one change each:
- [x] `backend/requirements.txt`: `edge-tts` 7.2.8 → 7.3.1 (old version gets 403), pin `av==16.1.0`
      (faster-whisper 1.1.1 breaks on PyAV 19)
- [x] `research/uxtd_transcriber/augment_with_full_vocab.py`: exclude v2/v3 words so a rebuild
      doesn't double-count 241 words
- [x] research scripts: paths now relative to the repo (`Path(__file__)`), replacing the hardcoded ones
- [x] `tasks/todo.md` (restore log) + `tasks/plan_history.md` + this file

## 5. Cleanups (nice to have)

- [x] Research scripts hardcode `/mnt/c/Users/rosha/Documents/...`. Replace with paths relative
      to the script (`Path(__file__).parent`) so anyone can run them.
- [ ] Name: the repo says ArticuPlay, the screens and code still say StoryWeaver. README now notes it; renaming not done.
- [x] MIT `LICENSE` added. Was: no `LICENSE` file for the code itself. Choose one (e.g. MIT for code, with the data/weights
      note kept separate).
- [ ] GitHub repo "About": add a one-line description and topics (`speech-therapy`, `whisper`,
      `lora`, `fastapi`, `react`, `childhood-apraxia-of-speech`).
- [ ] Back up the model files somewhere other than this PC (applies whether or not they go public).
