# Write-up (draft)

## External models, libraries and data (must stay complete until code freeze)

| Item | Used for | Runtime? | Licence |
|---|---|---|---|
| numpy | everything | yes | BSD-3 |
| onnxruntime | Silero VAD inference (optional) | optional | MIT |
| Silero VAD v5 (`models/silero_vad.onnx`) | speech/non-speech per frame (off by default) | optional | MIT |
| scipy, soundfile | synthetic data tools | no | BSD-3 / BSD-3 |
| scikit-learn | classifier training | no | BSD-3 |

No internet access is needed at runtime.

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

## Failure cases (≥ 5, with Arabic examples)
1. **GI-D03 "لحظة."** (car noise): one 450 ms word right at an agent phrase end, and the caller then goes silent. Acoustically it looks like a backchannel. Fix: words layer (ASR + stop lexicon: لحظة، لا، بس، اسمع).
2. **BG-D03 "يُمّه! وين الشاحن؟"** (failed in v1, passes in v2): the caller talks to someone in the room. v2 keeps talking only because this side talk is softer than the caller's own turn. A caller who shouts at the room would still trigger a stop. Real fix: words (addressee).
3. **OV-D02 "الوطنية، أيوه، معايا."** (failed in v1, passes in v2 by level only): a 2.4 s multi-word backchannel. Duration alone fires. A robust fix is a lexicon of acknowledgement phrases.
4. **GI-D02 "أيوه… … بس أبغى أغيّر الموعد"**: the intent is only evident at "بس". We stop late (≈0.9 s), because "أيوه" + pauses look like a backchannel.
5. **BG-D01 (TV speech)** during the agent's *first* utterance: the caller has not spoken yet, so there is no caller level to compare against and the TV can trigger a stop.
6. **Risk:** a caller who interrupts *softly* (tired, polite) is penalised by the level rule. The synthetic stress test with random interruption levels shows a big responsiveness drop.
