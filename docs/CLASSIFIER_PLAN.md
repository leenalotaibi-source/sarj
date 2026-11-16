# Backchannel-vs-interruption classifier: plan

## What it decides
After the first **300 ms** of a caller speech run during agent speech: is this
a genuine interruption (→ stop sooner) or a backchannel / background / noise /
echo leftover (→ wait longer)? Its output only **shifts** the rule-based
threshold (`clf_gain_ms`), so a bad model cannot make the system much worse
than v1. If it does not help on the official scorer, we turn it off.

Why a prefix: at runtime we only ever have the first N ms. Training on whole
clips would teach "is it long", which we cannot know when we must decide.

## Stage A (done, numpy-only at runtime): prosodic prefix features
`tt/prefix.py`: 12 scalar features. Level relative to the caller's own voice,
SNR, voicing, slope, residual echo, plus the agent context: phrase end, question,
pause, position in the utterance. The model is a logistic regression stored as
JSON. `tools/train_classifier.py` replays the **runtime** Decider over labelled
calls, so the training features match the live features exactly, including the
echo-suppression artefacts.

## Stage B: audio embeddings (only if A plateaus)
- Encoder: Whisper-tiny or Whisper-base encoder (MIT) or Ara-BEST-RQ (check licence) on the 300–500 ms caller prefix. Mean-pool and add a small MLP head.
- Must run in a background thread. Before depending on it, measure latency on our slowest laptop. Budget: < 150 ms per run on CPU.
- Same API as Stage A (`p_interrupt(prefix) -> float`), so the Decider does not change.

## Training data
| Source | Use | Licence check |
|---|---|---|
| Dev set (24 calls) | **evaluation only** (otherwise we overfit to the synthetic voices) | kit terms |
| SADA (SDAIA, Saudi dialects) | cut short turns (≤ 1–2 words: أيوه، إيه، تمام…) as BC, long turn starts as GI | check SADA terms for commercial use |
| Own consented recordings | BC / GI / laugh / cough, several speakers, Najdi + Hijazi, phone mic | consent form, ours |
| Multi-TTS (Gemini, SILMA, Habibi*, NAMAA*) | extra voices for agent + caller | *some NC: do not ship the data, only use if allowed |

All clips go through `tools/make_synth_calls.py`:
- Narrowband phone channel (≥ 50 %, matching the hidden set's ≥ 40 %).
- Line echo (random delay, gain and non-linearity).
- Noise at 8–35 dB SNR.
- Far-field reverb for background speech (BG).

## Evaluation protocol
1. Grouped CV by **speaker/clip** (the script does this) and AUC per category.
2. A/B on the official scorer: `use_clf=0` vs `use_clf=1` on all 24 dev calls, then on the D3 checkpoint.
3. Keep it only if it helps on **both**. Record the failure cases for the write-up.
