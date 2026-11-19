# Sarj hackathon: human-like turn taking (Arabic phone calls)

**v5, official real-time run on the 24 dev calls: Core 97.4 (R 0.973, S 0.969), end-of-turn 90.9, 0 protocol errors.**

While the agent speaks, decide every 20 ms: keep talking, `STOP`, or (optionally) `RESUME`.

## Layout

```
my_system.py              kit entry point (thin, never raises); all logic in tt/
tt/config.py              every tunable parameter (override with TT_CFG="k=v,k=v")
tt/frontend.py            per-frame features: echo suppression, noise floor, voicing, optional Silero VAD
tt/decider.py             run tracking + STOP/RESUME rules + agent-text cues
tt/words.py               words layer: offline Whisper tiny (transcript + keyword spotting) + Arabic lexicon
tt/eot.py                 end of turn (agent silent): pause, hesitation, words, digit count -> TAKE_TURN
tt/prefix.py              prefix features of a caller speech run + JSON logistic model
tools/augment.py          phone band / mu-law / echo / noise / far-field augmentation
tools/make_synth_calls.py build labelled stereo calls from clip folders
tools/replay.py           offline replay + APPROXIMATE score on our own synthetic calls
tools/train_classifier.py backchannel-vs-interruption classifier from labelled calls
tools/kit_eval.py         offline run of the 24 dev calls with the OFFICIAL scorer
tools/robustness.py       the same calls with the caller line degraded (gain, phone, noise, babble)
tools/site_data.py        refresh the demo page's per-call data from a real run
tools/build_site.py       embed tt/ + site/py/ + site/results.json into site/index.html
site/index.html           demo page: runs tt/ in the browser (Pyodide); no API keys in the source
site/worker.js            optional Cloudflare proxy that holds the Gemini key for the page
tests/smoke.sh            end-to-end plumbing test on fake audio (no kit/data needed)
models/silero_vad.onnx    Silero VAD v5 (MIT), only used with use_silero=1
```

## v1 pipeline

1. **Echo suppression (frontend).** Band powers between 100 and 3400 Hz, so wideband and phone audio look the same.
   - The echo delay is estimated by correlating the agent and caller energy envelopes.
   - The echo gain per band is a robust percentile of the caller/agent power ratio over the last ~4 s of agent speech.
   - Whatever caller energy is left above the estimated echo (+6 dB) is the "excess".
2. **Speech frame:** excess-over-noise SNR > 9 dB, plus voicing (pitch autocorrelation) or very high SNR.
3. **Background rejection:** runs more than 12 dB below the caller's own level (learned while the agent is silent) are ignored.
4. **Sustained-speech rule:** `STOP` after 520 ms of speech in one run (gaps up to 140 ms are tolerated). The threshold changes with context:
   - Shorter: the caller is louder than usual, or the agent just asked a question.
   - Longer: the run starts right after an agent phrase end, where backchannels are expected (from word timings).
   - Optional classifier: shifts the threshold by ±200 ms after the first 300 ms of the run.
5. **Optional soft stop:** `soft_stop=1` stops early and sends `RESUME` if the run dies within 450 ms.

## Commands

```bash
python -m ttbench.play --fast                          # official scorer, quick
TT_CFG="stop_after_ms=450,soft_stop=1" python -m ttbench.play --fast   # parameter sweep
TT_DEBUG=1 python -m ttbench.play --fast               # log every decision + slow frames
python -m ttbench.summary runs/<name>

bash tools/kit_eval.py         offline run of the 24 dev calls with the OFFICIAL scorer
tools/robustness.py       the same calls with the caller line degraded (gain, phone, noise, babble)
tools/site_data.py        refresh the demo page's per-call data from a real run
tools/build_site.py       embed tt/ + site/py/ + site/results.json into site/index.html
site/index.html           demo page: runs tt/ in the browser (Pyodide); no API keys in the source
site/worker.js            optional Cloudflare proxy that holds the Gemini key for the page
tests/smoke.sh                                    # plumbing test, no data needed
python tools/make_synth_calls.py --clips clips --out synth_out --n 200
python tools/replay.py synth_out --fails 10
python tools/train_classifier.py synth_out             # -> models/bc_clf.json, then use_clf=1
```

Every number from `tools/replay.py` is approximate. Only `ttbench` numbers go into the write-up.
