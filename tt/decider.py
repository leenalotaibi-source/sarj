"""Decision logic: turns per-frame speech evidence into STOP / RESUME.

Pure Python, no I/O, so it can be unit-tested and replayed offline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from .config import Config
from .frontend import Frontend, FrameFeatures
from .prefix import LinearModel, prefix_vector

STOP, RESUME = "STOP", "RESUME"
_PUNCT_END = re.compile(r"[.!?؟،,؛;:…]$")
_QUESTION = re.compile(r"[?؟]$")


@dataclass
class Word:
    text: str
    start_ms: float
    end_ms: float


@dataclass
class AgentPlan:
    """What we know about the agent utterance from agent_utterance_start."""
    text: str = ""
    words: List[Word] = field(default_factory=list)
    phrase_ends_ms: List[float] = field(default_factory=list)
    question_ends_ms: List[float] = field(default_factory=list)


def parse_agent_info(*args, **kwargs) -> AgentPlan:
    """Accept whatever the kit passes at agent start (dict, object, text, words)."""
    info = {}
    for a in args:
        if isinstance(a, dict):
            info.update(a)
        elif isinstance(a, str):
            info.setdefault("text", a)
        elif isinstance(a, (list, tuple)):
            info.setdefault("words", a)
        elif a is not None and hasattr(a, "__dict__"):
            info.update(vars(a))
    info.update(kwargs)
    plan = AgentPlan(text=str(info.get("text") or info.get("transcript") or ""))
    words = info.get("words") or info.get("word_timings") or info.get("timings") or []
    for w in words:
        try:
            if isinstance(w, dict):
                t = w.get("word") or w.get("text") or ""
                s = float(w.get("start_ms", w.get("start", 0)))
                e = float(w.get("end_ms", w.get("end", s)))
            else:
                t, s, e = str(w[0]), float(w[1]), float(w[2])
        except Exception:
            continue
        plan.words.append(Word(t, s, e))
    # timings in seconds? convert if the whole utterance is < 100 "units"
    if plan.words and plan.words[-1].end_ms < 100:
        for w in plan.words:
            w.start_ms *= 1000.0
            w.end_ms *= 1000.0
    for i, w in enumerate(plan.words):
        nxt = plan.words[i + 1] if i + 1 < len(plan.words) else None
        gap = (nxt.start_ms - w.end_ms) if nxt else 0
        if _PUNCT_END.search(w.text.strip()) or gap > 250:
            plan.phrase_ends_ms.append(w.end_ms)
        if _QUESTION.search(w.text.strip()) and nxt is not None:
            plan.question_ends_ms.append(w.end_ms)
    return plan


class Decider:
    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self.fe = Frontend(self.cfg)
        self.caller_ref_db = None   # caller's own speaking level (near-field)
        self.log = []               # (t_ms, event, details) for failure analysis
        self.agent_on = False
        self.plan = AgentPlan()
        self.prefix_sink = None     # training: list that receives (abs_ms, run_start_abs_ms, vec)
        self.prefix_ms = 300.0
        self.abs_t = 0.0
        self.clf = None
        if self.cfg.use_clf:
            try:
                self.clf = LinearModel(self.cfg.clf_path)
            except Exception:
                self.clf = None
        self._reset_utt()

    # --------------------------------------------------------------- events
    def _reset_utt(self):
        self.t = 0.0
        self.stopped = False
        self.stop_t = None
        self.resumed = False
        self.run_speech = 0.0
        self.run_gap = 1e9
        self.run_start = None
        self.run_levels = []
        self.last_agent_voice_t = 0.0
        self.run_in_pause = False
        self.run_frames = []
        self.run_ctx = (False, False, False, 0.0)
        self.clf_adjust = 0.0

    def agent_start(self, *args, **kwargs):
        self.agent_on = True
        try:
            self.plan = parse_agent_info(*args, **kwargs)
        except Exception:
            self.plan = AgentPlan()
        self._reset_utt()

    def agent_end(self, *args, **kwargs):
        self.agent_on = False
        self.stopped = False

    # ------------------------------------------------------------ main loop
    def frame(self, agent, caller) -> Optional[str]:
        cfg = self.cfg
        f = self.fe.process(agent, caller)
        self.t += cfg.frame_ms
        t = self.t
        self.abs_t += cfg.frame_ms

        if not self.agent_on:
            # the caller is the main speaker now: learn their near-field level
            if f.is_speech and f.snr_db > cfg.snr_strong_db:
                self.caller_ref_db = f.excess_db if self.caller_ref_db is None else \
                    0.97 * self.caller_ref_db + 0.03 * f.excess_db
            return None

        if f.agent_db > -50:
            self.last_agent_voice_t = t

        # ---- run tracking (speech frames with short gaps tolerated) ----
        speech = f.is_speech and t > cfg.min_ms_after_agent_start
        if speech and self._is_background(f):
            speech = False
        if speech:
            if self.run_gap > cfg.hangover_ms:
                self.run_start = t
                self.run_speech = 0.0
                self.run_levels = []
                self.run_in_pause = (t - self.last_agent_voice_t) > 150
                self.run_frames = []
                self.run_ctx = self.context(t)
                self.clf_adjust = 0.0
                self.run_start_abs = self.abs_t
            self.run_speech += cfg.frame_ms
            self.run_levels.append(f.excess_db)
            self.run_gap = 0.0
        else:
            self.run_gap += cfg.frame_ms
        if self.run_start is not None and self.run_gap <= cfg.hangover_ms:
            self.run_frames.append(f)
            n_ms = len(self.run_frames) * cfg.frame_ms
            if self.clf is not None and n_ms == self.clf.prefix_ms:
                p = self.clf.p_interrupt(prefix_vector(self.run_frames, self.caller_ref_db, *self.run_ctx))
                self.clf_adjust = -cfg.clf_gain_ms * (2 * p - 1)
            if self.prefix_sink is not None and n_ms == self.prefix_ms:
                self.prefix_sink.append((self.abs_t, self.run_start_abs,
                                         prefix_vector(self.run_frames, self.caller_ref_db, *self.run_ctx)))
        run_ended = self.run_gap > cfg.hangover_ms

        # ---- after a (soft) stop: maybe take it back ----
        if self.stopped:
            if (cfg.soft_stop and not self.resumed and run_ended
                    and self.run_speech < cfg.resume_if_run_under_ms
                    and t - self.stop_t <= cfg.resume_window_ms):
                self.stopped = False
                self.resumed = True
                self._log(t, RESUME, f)
                return RESUME
            return None

        if run_ended or self.run_start is None:
            return None

        need = self._needed_ms(f)
        if self.run_speech >= need:
            self.stopped = True
            self.stop_t = t
            self._log(t, STOP, f, need=need)
            return STOP
        return None

    # -------------------------------------------------------------- helpers
    def context(self, t: float):
        """(agent_pause, near_phrase_end, after_question, t_in_utt_ms) at time t."""
        return ((t - self.last_agent_voice_t) > 150,
                self._in_zone(t, self.plan.phrase_ends_ms, self.cfg.bc_zone_ms),
                self._in_zone(t, self.plan.question_ends_ms, self.cfg.question_zone_ms),
                t)

    def _is_background(self, f: FrameFeatures) -> bool:
        if self.caller_ref_db is None:
            return False
        return f.excess_db < self.caller_ref_db - self.cfg.bg_below_caller_db

    def _needed_ms(self, f: FrameFeatures) -> float:
        cfg = self.cfg
        if cfg.soft_stop and not self.resumed:
            need = cfg.soft_stop_after_ms
        else:
            need = cfg.stop_after_ms
            if self.resumed:
                need += 150.0  # we were wrong once this utterance; be stricter
        start = self.run_start
        if self.run_levels and self.caller_ref_db is not None:
            lvl = sum(self.run_levels) / len(self.run_levels)
            if lvl > self.caller_ref_db + cfg.loud_bonus_db:
                need -= cfg.loud_shorten_ms
        if self._in_zone(start, self.plan.phrase_ends_ms, cfg.bc_zone_ms):
            need += cfg.bc_zone_extend_ms
        elif not self.plan.phrase_ends_ms and self.run_in_pause:
            # no word timings: an agent pause is the fallback "phrase end"
            need += cfg.bc_zone_extend_ms
        if self._in_zone(start, self.plan.question_ends_ms, cfg.question_zone_ms):
            need -= cfg.question_shorten_ms
        need += self.clf_adjust
        return max(need, 160.0)

    @staticmethod
    def _in_zone(t: float, ends: List[float], width: float) -> bool:
        return any(0 <= t - e <= width for e in ends)

    def _log(self, t, ev, f, **kw):
        self.log.append((round(t), ev, repr(f), kw))
