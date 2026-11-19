"""Browser adapter (Pyodide): the live call on the demo page runs the REAL system (tt/), frame by frame.

The page streams the microphone as 20 ms frames, plus the agent's own voice as the echo reference.
Speech recognition cannot run in the browser, so `WebWords` stands in for the Whisper worker: when
the system asks for words, the page sends the same audio to a cloud recogniser and hands the text back.
Everything else (echo removal, caller level, STOP rules, end of turn) is the code that runs in the harness.
"""
import json

import numpy as np

from tt.config import Config
from tt.decider import Decider
from tt.eot import words_verdict
from tt.words import classify

FRAME = 320


class WebWords:
    """Same interface as tt.words.WordsWorker (submit / poll / busy), answered by the page."""
    TIMEOUT_MS = 4000

    def __init__(self):
        self.req = None
        self.audio = None
        self.mode = "stop"
        self.asked_at = 0.0
        self.ready = None

    @property
    def busy(self):
        return self.req is not None

    def submit(self, req_id, audio, now_ms, **ctx):
        if self.req is not None:
            return False
        self.req, self.audio, self.mode, self.asked_at = req_id, audio, ctx.get("mode", "stop"), now_ms
        return True

    def poll(self, now_ms):
        if self.ready is not None:
            r, self.ready, self.req, self.audio = self.ready, None, None, None
            return r
        if self.req is not None and self.audio is None and now_ms - self.asked_at > self.TIMEOUT_MS:
            self.req = None                  # the page never answered (offline, quota): acoustics only
        return None

    # ---- page side
    def take_audio(self):
        """Audio the system wants transcribed (int16 bytes), once; None if nothing is pending."""
        if self.req is None or self.audio is None:
            return None
        a, self.audio = self.audio, None
        return a.astype(np.int16).tobytes()

    def answer(self, text):
        if self.req is not None:
            self.ready = (self.req, text or "")


class Live:
    def __init__(self):
        self.cfg = Config()
        self.words = WebWords()
        self.use_words = True
        self.start()

    def start(self, use_words=True):
        self.use_words = bool(use_words)
        self.words = WebWords()
        self.dec = Decider(self.cfg, words=self.words if self.use_words else None)
        self.t = 0
        self.last_text = ""
        self.last_verdict = ""

    def agent_start(self, text, duration_ms):
        if self.words.req is not None and self.words.mode == "eot":
            self.words.req = self.words.audio = None      # the caller's turn is over: drop its words request
        self.dec.agent_start(str(text), float(duration_ms))

    def agent_end(self):
        self.dec.agent_end()

    # words: the page polls want_words(); if it returns a mode, it fetches words_audio() and later calls set_words()
    def want_words(self):
        return self.words.mode if (self.use_words and self.words.req is not None and self.words.audio is not None) else ""

    def words_audio(self):
        return self.words.take_audio()

    def set_words(self, text):
        mode = self.words.mode
        self.words.answer(text)
        self.last_text = text or ""
        if mode == "eot":
            self.last_verdict = "eot:" + (words_verdict(text, self.dec.eot.digits_needed) or "no objection")
        else:
            self.last_verdict = classify(text or "") or "none"
        return self.last_verdict

    def frames(self, agent_js, caller_js):
        a = np.frombuffer(agent_js.to_bytes(), dtype=np.int16)
        c = np.frombuffer(caller_js.to_bytes(), dtype=np.int16)
        n = len(c) // FRAME
        out, speech, level, ev = [], 0, -120.0, 0.0
        dec = self.dec
        for i in range(n):
            ai = a[i * FRAME:(i + 1) * FRAME]
            if len(ai) < FRAME:
                ai = np.zeros(FRAME, dtype=np.int16)
            r = dec.frame(ai, c[i * FRAME:(i + 1) * FRAME])
            self.t += 20
            ff = self._last
            if ff is not None:
                if ff.is_speech:
                    speech += 1
                level = max(level, ff.excess_db)
                if dec.agent_on and not dec.stopped and dec.run_start is not None and dec.run_gap <= dec.cfg.hangover_ms:
                    ev = max(ev, 100.0 * dec.run_speech / max(dec._needed_ms(ff), 1.0))
            if r is not None:
                out.append({"decision": r, "t": self.t})
        e = dec.eot
        return json.dumps({
            "d": out,
            "speech": speech,
            "level": round(float(level), 1),
            "ref": None if dec.caller_ref_db is None else round(float(dec.caller_ref_db), 1),
            "evidence": round(float(ev), 1),
            "talking": bool(dec.agent_on),
            "stopped": bool(dec.stopped),
            "verdict": dec.word_verdict or "",
            "eot_need": round(float(e.need_ms())) if not dec.agent_on and not e.sent else None,
            "eot_quiet": round(float(e.quiet_ms)) if not dec.agent_on and not e.sent and e.quiet_ms < 1e8 else None,
        })

    @property
    def _last(self):
        return getattr(self.dec.fe, "_tt_last", None)


def _install_capture():
    """Remember the last FrameFeatures on the frontend (for meters), without changing the system."""
    from tt import frontend
    if getattr(frontend.Frontend, "_tt_wrapped", False):
        return
    orig = frontend.Frontend.process

    def process(self, agent, caller):
        f = orig(self, agent, caller)
        self._tt_last = f
        return f
    frontend.Frontend.process = process
    frontend.Frontend._tt_wrapped = True


_install_capture()
live = Live()
