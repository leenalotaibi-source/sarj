"""Browser adapter (Pyodide): analyse an uploaded / recorded clip with the real system (tt/).

The clip is the caller's microphone only. Before `start_ms` the agent is silent (the caller answers the
greeting, which teaches the system this caller's level); from `start_ms` on the agent is "talking".
Words are not available here (no recogniser in the browser), so this is the acoustic system alone.
"""
import json

import numpy as np

from tt.config import Config
from tt.decider import Decider

FRAME = 320


def analyze(pcm_js, start_ms, dur_ms):
    import live  # noqa: F401  (installs the frame capture used below)
    x = np.frombuffer(bytes(pcm_js), dtype=np.int16)
    n = len(x) // FRAME
    dec = Decider(Config(), words=None)
    zeros = np.zeros(FRAME, dtype=np.int16)
    started = False
    decisions, speech, env, evid, rel = [], [], [], [], []
    first_speech = None
    peak, ev100, lvl100 = -120.0, 0.0, None
    for i in range(n):
        t = i * 20
        if not started and t >= start_ms:
            dec.agent_start("", max(1000.0, float(dur_ms) - start_ms))
            started = True
        r = dec.frame(zeros, x[i * FRAME:(i + 1) * FRAME])
        f = dec.fe._tt_last
        speech.append(1 if f.is_speech else 0)
        if started and f.is_speech and first_speech is None:
            first_speech = t
        peak = max(peak, f.caller_db)
        if started and dec.run_start is not None and dec.run_gap <= dec.cfg.hangover_ms and not dec.stopped:
            ev100 = max(ev100, 100.0 * dec.run_speech / max(dec._needed_ms(f), 1.0))
        if f.is_speech and dec.caller_ref_db is not None:
            v = f.excess_db - dec.caller_ref_db
            lvl100 = v if lvl100 is None else max(lvl100, v)
        if r is not None:
            decisions.append({"decision": r, "t": t + 20})
        if i % 5 == 4:
            env.append(round(float(peak), 1))
            evid.append(round(float(ev100), 1))
            rel.append(None if lvl100 is None else round(float(lvl100), 1))
            peak, ev100, lvl100 = -120.0, 0.0, None
    return json.dumps({
        "duration_ms": n * 20,
        "decisions": decisions,
        "enrolled_db": None if dec.caller_ref_db is None else round(float(dec.caller_ref_db), 1),
        "first_speech": first_speech,
        "threshold": 100,
        "env": env,
        "evidence": evid,
        "rel": rel,
        "speech": speech,
    })
