"""End-of-turn layer: while the AGENT is silent, decide when the caller has finished.

Separate leaderboard from the Core score; decisions here never touch the Core
path (the scorer ignores TAKE_TURN for Core, and this module only runs while
the agent is not speaking).

A pause is "the end" only if nothing says the caller is still going:

    acoustics   the last sound before the pause was a hesitation: a hummed
                «اممم…» (little spectral change) or a syllable held on a
                steady pitch («المغرب…»)               -> wait long
    words       (same Whisper as the barge-in layer, one request per pause)
                ends on a connector or filler («و»، «يعني»، «بس»)   -> wait
                asked the agent to wait («لحظة أشوف»)            -> wait long
                is reading out digits and has not reached the count the
                agent asked for («هو سبع أرقام»)                   -> wait
                has read out all the digits the agent asked for     -> answer sooner
    rhythm      without words: each time the caller comes back after a long
                pause, they are thinking in chunks, so be a little more
                patient for the rest of this turn

Times in ms. Every threshold is in Config (eot_*), so the scorer sweep can move them.
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np

from .words import normalize

TAKE_TURN = "TAKE_TURN"

# the answer is not finished if its last word is one of these
CONTINUERS = {
    "و", "او", "ولا", "بس", "يعني", "لكن", "بعدين", "ثم", "عشان", "علشان", "لان", "اذا", "لو", "اللي", "الي",
    "في", "على", "من", "مع", "الى", "عند", "حق", "تقريبا", "مثلا", "يمكن", "ان", "انه", "كذا", "هو", "هي",
    "انا", "احنا", "رقم", "الساعه", "يوم",
}
FILLERS = {"ام", "امم", "مم", "اه", "اا", "اممم", "همم", "يعني", "شي", "وش", "ايش", "شلون", "اقول", "هذا", "ذا"}
# the caller asked the agent to wait while they look something up
WAIT_WORDS = {"لحظه", "لحظة", "ثانيه", "ثواني", "دقيقه", "استنى", "استني", "انتظر", "انتظري", "اصبر", "اشوف",
              "اشيك", "خلني", "خليني", "ادور", "افتح", "اتاكد"}
DIGIT_WORDS = {
    "صفر": 0, "واحد": 1, "وحده": 1, "اثنين": 2, "ثنين": 2, "اثنان": 2, "ثلاثه": 3, "ثلاث": 3, "تلاته": 3,
    "اربعه": 4, "اربع": 4, "خمسه": 5, "خمس": 5, "سته": 6, "ست": 6, "سبعه": 7, "سبع": 7, "ثمانيه": 8,
    "ثمان": 8, "ثمنيه": 8, "تسعه": 9, "تسع": 9,
    # dialect spellings and common recogniser spellings
    "تلاته": 3, "تلات": 3, "تلاث": 3, "تلاثه": 3, "ثلاته": 3, "اتنين": 2, "تنين": 2, "تمانيه": 8, "تمنيه": 8,
    "ثمنه": 8, "زيرو": 0,
}
_COUNT_ASKED = re.compile(r"(\S+)\s+(?:أرقام|ارقام|خانات|رقم|digits)")
_COUNT_WORDS = {"ثلاث": 3, "ثلاثة": 3, "أربع": 4, "اربع": 4, "أربعة": 4, "خمس": 5, "خمسة": 5, "ست": 6, "ستة": 6,
                "سبع": 7, "سبعة": 7, "ثمان": 8, "ثمانية": 8, "تسع": 9, "تسعة": 9, "عشر": 10, "عشرة": 10}
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def digits_asked(agent_text: str) -> Optional[int]:
    """«هو سبع أرقام» -> 7: how many digits the agent asked the caller to read out."""
    for m in _COUNT_ASKED.finditer(agent_text or ""):     # «رقم الطلب … هو سبع أرقام»: the first number word wins
        w = m.group(1).translate(_ARABIC_DIGITS)
        if w.isdigit():
            return int(w)
        if w in _COUNT_WORDS:
            return _COUNT_WORDS[w]
    return None


def _edit1(a: str, b: str) -> bool:
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) > len(b):
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i + (len(a) == len(b)):] == b[i + 1:]


def is_digit_word(w: str) -> bool:
    if w.startswith("و") and len(w) > 3:
        w = w[1:]
    # the recogniser slips on spoken digits («سفر» for «صفر»): one edit allowed on 3+ letter words
    return w in DIGIT_WORDS or (len(w) >= 3 and any(_edit1(w, d) for d in DIGIT_WORDS if len(d) >= 3))


def count_digits(text: str) -> int:
    """Digits read out so far, written as words («ثلاثة، سبعة») or as numerals («37»)."""
    t = (text or "").translate(_ARABIC_DIGITS)
    return sum(ch.isdigit() for ch in t) + sum(is_digit_word(w) for w in normalize(t).split())


_HUM = re.compile(r"^[اهمن]*[مه][اهمن]*$")


def is_filler(w: str) -> bool:
    """«امم»، «اه»، and how a small recogniser writes a hum («امام»، «هم»)."""
    return w in FILLERS or (len(w) <= 4 and bool(_HUM.match(w)) and w not in {"ما", "من", "انا", "هنا"})


def words_verdict(text: str, digits_needed: Optional[int]) -> str:
    """'wait_long' | 'wait' | 'complete' | '' (no objection) for the caller's words so far.

    The words only ever make us MORE patient, with one exception: the caller
    has read out as many digits as the agent asked for. A small recogniser
    gets Arabic words wrong often enough that "sounds finished" is not trusted."""
    toks = [w for w in normalize(text).split() if len(w) > 1 or w == "و"]   # stray letters are recogniser noise
    if not toks:
        return ""
    digits = count_digits(text)
    if digits_needed and digits >= 2:
        if digits >= digits_needed:
            return "complete"
        # one digit short: the recogniser often drops one, so no objection (normal wait) rather than "wait"
        return "" if digits_needed >= 5 and digits == digits_needed - 1 else "wait"
    last = toks[-1]
    if any(w in WAIT_WORDS for w in toks[-3:]):
        return "wait_long"
    if last in CONTINUERS or is_filler(last) or (last.startswith("و") and len(last) <= 2):
        return "wait"
    if digits >= 2 and sum(is_digit_word(w) for w in toks[-3:]) >= 2:
        return "wait"          # reading out a number of unknown length: pauses between groups
    return ""


class EndOfTurn:
    def __init__(self, cfg):
        self.cfg = cfg
        self.digits_needed = None
        self.reset()

    def reset(self):
        self.spoke_ms = 0.0          # caller speech in this turn
        self.quiet_ms = 1e9
        self.sent = True             # nothing to answer until the caller speaks
        self.comebacks = 0
        self.pitches = []            # pitch of the latest voiced frames
        self.fluxes = []
        self.hesitating = False
        self.words = ""              # transcript of the turn so far (latest request)
        self.words_for = -1          # pause number the transcript belongs to
        self.pause_no = 0
        self.turn_start_abs = None

    def agent_ended(self, agent_text: str):
        """The agent just finished (or was stopped): a new caller turn may start."""
        self.reset()
        self.digits_needed = digits_asked(agent_text)

    # ------------------------------------------------------------------ main
    def frame(self, f, abs_t: float, caller_ref_db: Optional[float]) -> Optional[str]:
        cfg = self.cfg
        voice = f.is_speech and (caller_ref_db is None or f.excess_db >= caller_ref_db - cfg.eot_bg_db)
        if voice:
            if self.turn_start_abs is None:
                self.turn_start_abs = abs_t
            if self.spoke_ms > 0 and cfg.eot_comeback_pause_ms <= self.quiet_ms < 1e8:
                self.comebacks += 1
            self.spoke_ms += cfg.frame_ms
            self.quiet_ms = 0.0
            self.sent = False
            self.fluxes = (self.fluxes + [f.flux])[-15:]
            self.pitches = (self.pitches + [f.pitch_hz if f.voicing > 0.45 else 0.0])[-30:]
            return None
        if self.sent or self.spoke_ms < cfg.eot_min_turn_ms:
            if self.quiet_ms < 1e8:
                self.quiet_ms += cfg.frame_ms
            return None
        if self.quiet_ms == 0:
            self.pause_no += 1
            self.hesitating = self._hum() or self._held_pitch()
        self.quiet_ms += cfg.frame_ms
        if self.quiet_ms >= self.need_ms():
            self.sent = True
            return TAKE_TURN
        return None

    def need_ms(self) -> float:
        cfg = self.cfg
        v = self.verdict()
        if v == "complete":
            return cfg.eot_done_ms
        if self.words_for == self.pause_no and self.words.strip():
            need = cfg.eot_quiet_ms          # the words were read and raise no objection
        else:                                # no words (yet): the caller's rhythm is all we have
            need = cfg.eot_quiet_ms + min(cfg.eot_patience_max_ms, cfg.eot_patience_step_ms * self.comebacks)
        if self.hesitating:
            need = max(need, cfg.eot_hesitation_ms)
        if v == "wait":
            need = max(need, cfg.eot_wait_ms)
        elif v == "wait_long":
            need = max(need, cfg.eot_wait_long_ms)
        return need

    def verdict(self) -> str:
        return words_verdict(self.words, self.digits_needed) if self.words_for == self.pause_no else ""

    # -------------------------------------------------------------- acoustics
    def _hum(self) -> bool:
        """«اممم…»: a sustained sound with almost no spectral change."""
        fl = self.fluxes[-10:]
        return len(fl) >= 10 and float(np.mean(fl)) < self.cfg.eot_hum_flux

    def _held_pitch(self) -> bool:
        """A final syllable held on one pitch («المغرب…»): the caller is still thinking."""
        tail = [p for p in self.pitches if p > 0]
        need = int(self.cfg.eot_held_ms / self.cfg.frame_ms)
        if len(tail) < need:
            return False
        tail = np.log2(np.array(tail[-need:]))
        d = np.abs(tail - np.median(tail))
        d = np.minimum(d % 1.0, 1.0 - d % 1.0)       # octave errors of the pitch tracker are not changes
        return float(np.mean(d <= np.log2(1 + self.cfg.eot_held_spread))) >= 0.8

    # ------------------------------------------------------------------ words
    def wants_words(self) -> bool:
        """At the first frame of each pause: ask the recogniser what the caller said."""
        return (not self.sent and self.quiet_ms == self.cfg.frame_ms and self.spoke_ms >= self.cfg.eot_min_turn_ms
                and self.turn_start_abs is not None)

    def set_words(self, pause_no: int, text: str):
        if pause_no == self.pause_no:
            self.words, self.words_for = text, pause_no
