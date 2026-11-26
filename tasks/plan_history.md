# ArticuPlay (StoryWeaver V2) — Plan History

Reconstructed 2026-11-26 from the original Claude Code chats, so the plan can be followed without
them:

- March 2026 planning chat (in the PIRS project folder, this session's history)
- 2026-03-25 StoryWeaver V2 scaffold chat (`.claude/projects/C--Users-rosha-Documents-StoryWeaver`)
- 2026-07-25 → 07-29 build chats in `Documents\sw_2` (`.claude/projects/C--Users-rosha-Documents-sw-2`,
  two near-identical transcripts plus one short one)
- `tasks/todo.md`, `tasks/lessons.md` and the README in this repo

Where a chat and the code disagree, the code is treated as the truth. Dates are 2026.

---

## 1. Starting point: StoryWeaver V1 (2025, college)

Streamlit app for children aged 4–6. The child describes a picture, Whisper transcribes it, Gemini
finds grammar, vocabulary and articulation errors, Gemini writes a practice story around the
difficult words, and the child reads it line by line with gTTS model audio.
Pilot: 2 children, 6 sessions, 88% transcription accuracy, practice accuracy 72% → 87%.

## 2. First V2 plan (March, before the professor meeting)

Bigger version of the same idea:

- Ages 4–6, 14–15 minute sessions: warm-up mascot → picture prompt → error detection (9 ASHA error
  types: omission, substitution, addition, distortion, syllable deletion, inconsistent production,
  grammar, vocabulary, fluency) → modeling phase → story generation with Indian cultural context →
  line-by-line practice with fading cues → badges → parent tip → save.
- Stack: FastAPI + Supabase, React web first, React Native later, Whisper API then a fine-tuned
  Whisper, Gemini, ElevenLabs voices.
- Research: fine-tune Whisper on Indian children's speech, pilot with 20+ children, Interspeech 2026.
- A backend scaffold was written on 2026-03-25 around stories (`TherapySession`, `SessionLine`,
  `ErrorReport`). It was later replaced.

## 3. Professor's feedback (March) and the redesign

The professor, a speech researcher, asked for:

1. One specific group of people with a specific disability.
2. Work out what help they need, then shape the app around it.
3. Keep it simple.
4. Make it interactive: the app says something, the child says it back, the app checks and corrects.
   Single words up to one sentence at most. No long stories, they take too long.
5. Offer more than one language.

He also asked whether a dataset existed, which led to the dataset research below.

### Choice of disability: Childhood Apraxia of Speech (CAS)

| Condition | Fit | Reason |
|---|---|---|
| CAS | Best | Standard CAS therapy (DTTC) is hear → repeat → precise feedback → retry, which is exactly the app's loop |
| Dyslexia | Poor | A reading disorder, needs visual and letter work, not speech production |
| Autism (speech delay) | Moderate | Too varied; many children need touch cues or AAC an app cannot give |

CAS affects 1–2 children per 1,000, is under-diagnosed, few apps target it, and speech therapists
are scarce in India.

### Redesigned product

- Children aged 4–10 with CAS; parents secondary; therapists later.
- Core loop: app speaks a word → child records → Whisper transcribes → `phonemizer` turns both into
  IPA sounds → compare → name the error and where it is → feedback → retry (up to 3) or next.
- Five levels:
  1. one syllable (cat, cup, sun)
  2. two syllables (rabbit, butter)
  3. three syllables (elephant)
  4. short phrase (big red ball)
  5. one sentence maximum (I see a cat.)
- Level unlocks automatically at 80% accuracy or better across 3 sessions.
- 10–12 minute session: mascot greets the child by name, 8–10 drill items, then stars, streak, a
  parent tip from Gemini, and everything saved.
- Screens: login, children list, add child, drill screen (the main work), session complete, progress.

### Languages

The first draft proposed English plus Hindi, Tamil and Telugu. You asked why only Indian
languages and asked to keep Arabic and add Chinese. Final plan:

| Language | Phase | Why |
|---|---|---|
| English | 1 (first) | Builds on V1, easiest to test |
| Arabic | 2 | CAS barely studied in Arabic; Gulf market |
| Hindi | 2 | Severe shortage of speech therapists in India |
| Mandarin | 2 | Tonal language, so CAS assessment is a different research problem |

The idea was that language would be a setting, not a separate app. The backend and pipeline stay
the same. Per language you would pass Whisper's language code, switch the edge-tts voice
(`en-US-AnaNeural`, `ar-SA-ZariyahNeural`, `hi-IN-SwaraNeural`, `zh-CN-XiaoxiaoNeural`), switch
phonemizer's language, add a word list, and translate the screens with `react-i18next`.
Target: 50 words × 5 levels × 4 languages = 1,000 drill items, English first.

Research angle: comparing CAS error patterns across four very different languages would be a paper
of its own, and collecting Arabic and Hindi child speech through the app would create the first
such dataset.

### Constraints you set

- Solo build, 4–6 weeks to a working prototype, research and product together.
- 100% free tools: self-hosted Whisper instead of the paid API, edge-tts instead of ElevenLabs,
  Gemini free tier, Supabase free tier, Railway and Vercel free tiers.
- "Not training any model; all tools used as they are." (This changed in July, see section 5.)

### Expected detection coverage

Text comparison could catch about 70% of CAS error types: substitution, omission, addition,
syllable deletion, cluster reduction, and inconsistency across attempts. It could not catch
distortions (such as a lisp) or prosody (stress and rhythm), since Whisper returns text and not
sound quality. Those were left for a later phase with `librosa` or `parselmouth`.

### Datasets (the professor's question)

- No public audio dataset of children with CAS exists anywhere.
- For testing: UltraSuite (children with speech sound disorders, free, non-commercial licence) and
  MyST (English child speech).
- Other languages: almost nothing for Arabic or Hindi; Mandarin has ChildMandarin (ages 3–5).
- Plan: the app records every attempt, so a pilot with 20+ children would itself become the dataset.

### Six-week plan

| Week | Goal |
|---|---|
| 1 | Supabase, FastAPI skeleton, Whisper and TTS endpoints, English word list |
| 2 | `compare_ipa()` and the full drill loop, tested without a UI |
| 3 | React frontend: login, child profiles, drill screen |
| 4 | Retry logic, session summary, parent tips, progress graph, auto-levelling |
| 5 | Test with 2–3 real children, fix bugs, under 3 s per item |
| 6 | Deploy to Railway and Vercel, share the link, demo to the professor, draft ethics approval |

Research: Paper 1 on English CAS feedback (Interspeech 2026), Paper 2 cross-language (2027).

---

## 4. The build (25–29 July, in `Documents\sw_2`)

### Week 1–2: backend
- Windows Smart App Control blocked Python libraries, so development moved to WSL2 Linux.
- Early checks: phonemizer worked; Whisper `base` was the smallest model that didn't hallucinate on
  child speech.
- You asked that a parent can add several children. Built.
- Built `compare_ipa()` (edit-distance alignment of sounds, classifies substitution, omission,
  addition, cluster reduction, syllable deletion), plus scoring, session completion, and per-sound
  logging.

### The decision that changed the project: train your own model
At first the app gave Whisper the target word as a hint. You called that "kind of cheating": the
app must notice when a child says a different word. You then asked for a model of your own,
"no shortcut", so the project would stand out. Your goal answer was "both: research now, company
later". Decisions made, in your words where recorded:
- Train on UltraSuite. Start with Whisper-small, not medium.
- Your camera idea (a camera watching the child's mouth for visual articulation feedback): "a new
  idea, not anywhere"; added as a later phase, not built.
- "Take the risk, what if results are better?": unfreeze the encoder in the classifier phase.
- On 26 July: "I don't want my app to guess and make it right what the child is saying. I want my
  app to be truthful, say what it heard, and then correct the child." This became the rule behind
  removing the target-word hint.

### Research phases
- **Phase 1, sound classifier:** UltraSuite has 864 recordings where a speech therapist scored one
  k, g or r sound. Trained a classifier on them: 60.9% with the encoder frozen, 67.9% after
  unfreezing it with LoRA.
- **Phase 2, transcriber:** fine-tuning on the disordered recordings was dropped because their
  labels are the word the child was asked to say, not what they said, which would teach the model
  to hide errors. Fine-tuned on typical children's recordings (UXTD) instead.
- **Fixes along the way:** 66% of the UXTD labels were session instructions, not speech (removed
  with a hand-built list); added synthetic sentences; beam search; trained on the app's own words;
  expanded to 491 words; built a proper held-out test.
- **Final result:** 96.0% vs 80.0% on 30 words never seen in training, repeated with 3 seeds.

### Week 3: frontend
React with TypeScript, plus a register screen (missing from the original plan). You tested the full
loop with your own voice on 26 July.

### Integration into the app
Fine-tuned model merged and converted to run in the backend, the target-word hint removed, and the
phone classifier added as a check that can only change a pass into a fail. On disordered recordings:
false accepts 28% → 15% → 3%, false rejects about 46%. Lowering the pass threshold was tried; it
swapped one error for the other.

### Publishing and security (29 July)
- GitHub repo `roshnrf/ArticuPlay` (public), README with all 12 model versions and an archive of old
  scripts.
- The name ArticuPlay was chosen for the repo; the code and screens still say StoryWeaver.
- Security audit: session endpoints had no login check (fixed); Supabase tables were open through
  its auto-generated API (row-level security enabled).
- LinkedIn posts drafted about the process and its failures.

---

## 5. Plan vs reality

| Planned | State |
|---|---|
| CAS focus, word-to-sentence drill loop | Built |
| Login, several children per parent, drill screen, session summary | Built |
| Error types: substitution, omission, addition, syllable deletion, cluster reduction | Built |
| Inconsistency detection | Not built (the data is stored; nothing reads it) |
| Distortion and prosody | Out of scope, as planned |
| English word list, 250 | Built, expanded to 491 |
| Stars and streak | Built |
| Auto-levelling (80% over 3 sessions) | **Not built.** New children start at level 1 and the screen offers no way to change it |
| Parent tips (Gemini), progress graph | Not built |
| Arabic, Hindi, Mandarin | Not started; only the 4 TTS voices are in code |
| "No model training" | Reversed: fine-tuned Whisper plus a sound classifier |
| Week 5 test with real children | Not done |
| Week 6 deploy and share a link | Not done |
| Ethics approval, UltraSuite licence email | Not done |
| Camera-based feedback | Idea only |

## 6. What adding a language now involves

Your July note: each language needs its own LoRA adapter trained on that language's audio, plus the
phonemizer language switch. The practical blocker is data. UltraSuite is English only, and public
child speech for Arabic and Hindi is close to none. Adding a language means finding or recording
child speech first, then a word list, a held-out test, and translated screens.

## 7. Restored 2026-11-16

The July folder and the WSL install were gone; GitHub was the only copy. Rebuilt everything from it
(see `todo.md`). Retrained model: 96.7% vs 86.7% on the same 30 held-out words. App runs locally.
