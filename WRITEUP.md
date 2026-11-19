# Write-up (draft)

## External models, libraries and data (must stay complete until code freeze)

| Item | Used for | Runtime? | Licence |
|---|---|---|---|
| numpy | everything | yes | BSD-3 |
| onnxruntime | Silero VAD inference (optional) | optional | MIT |
| Silero VAD v5 (`models/silero_vad.onnx`) | speech/non-speech per frame (off by default) | optional | MIT |
| scipy, soundfile | synthetic data tools | no | BSD-3 / BSD-3 |
| scikit-learn | classifier training | no | BSD-3 |
| Whisper tiny (OpenAI), CTranslate2 conversion `Systran/faster-whisper-tiny` | words layer: caller ASR (optional, `use_words=1`) | optional | MIT |
| faster-whisper + CTranslate2 | Whisper inference on CPU, int8 | optional | MIT |

| Pyodide 0.26 (MPL-2.0), Google Gemini (optional, own key) | demo page only: Python in the browser; agent replies, voice and live transcription | demo only | – |

No internet access is needed at runtime. The demo page (`site/`) is not part of the submitted system.

## Key design findings (dev set, offline replay with the official scorer)

| Step | Core | What changed and why |
|---|---|---|
| Kit baseline (any sound) | 41.0 | stops at every sound |
| v1: echo-aware gate + 520 ms rule | 57.2 | R 0.90 but S 0.37: backchannels, TV, horns all trigger |
| Noise floor from raw caller channel | ↑ | car noise was being absorbed into the echo estimate and erased real words ("لحظة") |
| Robust caller level (80th pct of own turn) | ↑ | a running average was dragged down by onsets and background |
| Graded "softer than the caller" penalty | 92.2 | interruptions come at full voice (0 to −4 dB vs own turn); backchannels, TV, side talk are 4–14 dB softer |
| Leaky speech accumulation | ↑ | "لا … لا … لحظة": short insistent words separated by pauses must add up |

**Official real-time run, all 24 dev calls (`ttbench.play --calls all`, speed 1.0): Core 92.4 (R 0.87, S 0.97), 0 protocol errors, max send lag 14.7 ms.**

Plateau, not a peak: Core stays 84–92 for caller-level percentile 70–90 and stop threshold 440–600 ms.

## Generalisation check against the team's P3 (offline replay, official scorer)

| System | dev (24) | xdev (25, team's dev split) | holdout (25, frozen) |
|---|---|---|---|
| P3 hybrid (Silero + caller-relative, b32cfc2) | 73.4 | 95.8 | 82.0 |
| v2 (this branch) | **92.2** | 68.5 | **85.4** |

- P3 stops after 200 ms of normal-level speech: it catches one-word commands (توقف، لا) but also stops on clear backchannels ("أها" in BC-D02; BC restraint 0.20 dev / 0.12 holdout).
- v2 waits ~520 ms and longer for softer speech or speech at agent phrase ends: backchannels pass, but isolated one-word commands in xdev ("توقف", 0.5 s) are missed.
- In xdev, a single "نعم" is KEEP and a single "توقف" is STOP. No acoustic cue separates them, so this needs the words layer (keyword spotting).

## Words layer: is it worth it? (oracle upper bound)

`tt/words.py` adds streaming ASR on the caller run, with an Arabic lexicon:
- **stop:** لا، لحظة، توقف، بس، اسمع، إلغاء… → STOP at once.
- **keep:** نعم، إيه، أيوه، أها، تمام… → wait 800 ms longer.

To size the gain before choosing a model, `tools/kit_eval.py --oracle-words D` feeds a *perfect* transcript from the labels, each word released only after it has been spoken plus a delay of D ms. This is **analysis only**: labels are never used by the system.

| ASR delay D | dev | xdev |
|---|---|---|
| no words | 92.2 | 68.5 |
| 300 ms | 98.3 | 88.2 |
| 600 ms | 96.1 | 82.5 |
| 1000 ms | 93.4 | 70.9 |

The gain depends on latency, not just accuracy: a model must return in ≲ 500 ms on the laptop CPU. Candidate: faster-whisper base/tiny, int8 (MIT). This still has to be measured with the real model; Whisper on 0.5–1 s clips may also hallucinate.

### Real model (faster-whisper tiny, int8, CPU), delay included in the score

| System | dev (offline) | dev **real-time** (`ttbench.play --calls all`) | xdev | holdout |
|---|---|---|---|---|
| P3 (team) | 73.4 | 73.41 (team's run) | 95.8 | 82.0 |
| v2 acoustic | 92.2 | 92.4 | 68.5 | 85.4 |
| v3a words, token = +800 ms | 94.4 | 90.7 | 83.9 | 84.3 |
| v3 words, token = run must outlast a backchannel | 96.2 | 96.4 | 84.0 | **87.1** |
| **v4 = v3 + Whisper keyword spotting** | **97.5** | **97.4** | **87.0** | 85.3 |

Holdout was scored three times (v3a, v3, v4), and no parameter was chosen from it. Real-time v3: 0 protocol errors, max send lag 31 ms.

Lessons:
- **Latency beats accuracy.** Whisper base is more accurate, but at 500–700 ms per request its words arrive too late (dev 92.2, xdev 69.6). tiny takes ~300 ms.
- **Arabic answers start with a listener token** ("إيه الأحد … زين، احجزه", "أيوه… بس أبغى"). If a token simply adds waiting time (v3a), real interruptions get delayed, and real time dropped to 90.7. Instead, a token only asks the run to outlast a typical backchannel (800 ms of speech).
- Whisper on short clips needed three fixes:
  1. No initial prompt: it copied the prompt words into the output.
  2. A 12-token cap: it looped "لا، لا، لا…".
  3. One-edit fuzzy matching for distinctive command words ("لاحظة" → لحظة). "ماشي" must not become "مابي".
- Background voices say stop words too (BG-D53: "بس" at −7.5 dB). A stop word only counts at near the caller's own level.
- If the model folder is missing, the system runs on acoustics alone (test: `tests/test_my_system.py`).

### v4: keyword spotting for isolated commands ("توقف")

Free transcription fails on one isolated word: Whisper tiny writes "توقف" as "تبقص" or "لطوق في الغر", and "لا إلغاء" as "لأكيدها".

Instead, v4 asks the same model how likely each candidate word is, by forced alignment on the same encoding:
- stop candidates: لا، توقف، لحظة، إلغاء، اسمع
- listener candidates: نعم، إيه، أيوه، أها، طيب، تمام

If the best stop word beats the best listener word by ≥ 0.3 (mean log-probability), and scores ≥ −2.5, it counts as heard.

| Variant (dev + xdev: 18 STOP / 32 KEEP events) | time / request | stop words caught | KEEP heard as stop |
|---|---|---|---|
| free transcript only (v3) | ~330 ms | 8/18 | 0/32 |
| + keywords, 30 s window | ~950 ms | 13/18 | 0/32 |
| **+ keywords, 10 s window, 11 candidates (v4)** | **~190 ms** | **12/18** | **0/32** |

- **Faster:** Whisper normally pads every clip to 30 s. Encoding only a 10 s window cuts encoding from ~350 ms to ~40 ms with almost no loss on 1–3 s clips. v4 is faster than v3 even though it does more.
- **Closed-set trap:** a word missing from the candidate list is matched to its nearest candidate. "طيب" (AM-D01) was scored as "توقف" until "طيب" and "تمام" were added as listener candidates. Common listener words must be in the list.
- **Holdout:** 87.1 → 85.3. Responsiveness rose (0.919 → 0.944), but one background-speech event became a false stop. Not tuned further on holdout.

## v5: end of turn (own leaderboard)

While the agent is silent, `tt/eot.py` decides when the caller has finished (`TAKE_TURN`). It never touches the STOP path.

- Base rule: answer after **700 ms** of quiet (full promptness credit lasts 600 ms after the turn end).
- **Hesitation** (acoustics): the last sound was a hum (little spectral change) or a syllable held on one pitch → wait 2 s.
- **Words** (same Whisper tiny, one request per pause, longer transcript, no keyword spotting):
  - ends on a connector or filler («و»، «يعني»، «بس»، «امم») → wait 1.6 s;
  - asked the agent to wait («لحظة أشوف») → wait 3 s;
  - reading out digits → wait, **unless** the caller has read as many digits as the agent asked for: «هو سبع أرقام» in the agent's text → 7 → answer after 450 ms.
- Without words: each comeback after a long pause adds 250 ms of patience (max 1 s).

**Words only ever add patience** (except the digit count). Whisper tiny writes «بعد المغرب يعني تقريباً الساعة سبع» as «بحدا مغرب! يأني! تقليبنس!»: good enough to spot «يعني» or «لحظة», not to decide that an answer "sounds finished".

| System | End-of-turn (dev, 2 events, 5 required pauses) |
|---|---|
| Kit baseline (700 ms of quiet) | 28.5 |
| TurnSense v3.1 (team page, offline) | 78.8 |
| v5, acoustics only (`use_words=0`) | 64.7 |
| **v5, real time** | **90.9** (P 0.83, Q 1.00) |

Core in the same real-time run: **97.4** (R 0.973, S 0.969), 1 false stop, 0 misses, 0 protocol errors (`experiments/v5-rt/`). Two events are far too few to tune on: the thresholds are round numbers chosen from the scoring rule, not from these calls.

## Robustness (`tools/robustness.py`, offline, caller line degraded)

| Condition | Baseline | v5 |
|---|---|---|
| As is | 41.0 | 96.5 |
| 6 dB quieter | 52.0 | 94.7 |
| 6 dB louder | 34.0 | 91.7 |
| Narrowband phone | 47.8 | 89.8 |
| Noise, SNR 20 dB | 6.2 | 95.5 |
| Noise, SNR 10 dB | 0.0 | 83.6 |
| Babble, 15 dB | 0.0 | 52.6 |
| Phone + quieter + noise 20 dB | 44.0 | 92.5 |

Gain changes barely matter (every level is relative to the caller). **Babble is the weak spot**: other voices at a caller-like level pass the level rule (S 0.33). Fix: a speaker embedding of the caller from the greeting.

## Failure cases (≥ 5, with Arabic examples)
1. **GI-D03 "لحظة."** (car noise): one 450 ms word right at an agent phrase end, and the caller then goes silent. Acoustically it looks like a backchannel. Fix: words layer (ASR + stop lexicon: لحظة، لا، بس، اسمع).
2. **BG-D03 "يُمّه! وين الشاحن؟"** (failed in v1, passes in v2): the caller talks to someone in the room. v2 keeps talking only because this side talk is softer than the caller's own turn. A caller who shouts at the room would still trigger a stop. Real fix: words (addressee).
3. **OV-D02 "الوطنية، أيوه، معايا."** (failed in v1, passes in v2 by level only): a 2.4 s multi-word backchannel. Duration alone fires. A robust fix is a lexicon of acknowledgement phrases.
4. **GI-D02 "أيوه… … بس أبغى أغيّر الموعد"**: the intent is only evident at "بس". We stop late (≈0.9 s), because "أيوه" + pauses look like a backchannel.
5. **BG-D01 (TV speech)** during the agent's *first* utterance: the caller has not spoken yet, so there is no caller level to compare against and the TV can trigger a stop.
6. **HT-D02 «ثلاثة، سبعة، تسعة … واحد، اثنين …»** (end of turn): we took the turn once at 19.5 s, in a gap inside the dictation that is not labelled as a pause. The digit count then held us correctly until the seventh digit. Fix: count digit groups acoustically too, so we do not depend on the recogniser hearing every digit.
7. **Risk:** a caller who interrupts *softly* (tired, polite) is penalised by the level rule. The synthetic stress test with random interruption levels shows a big responsiveness drop.
