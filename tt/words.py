"""Words layer: what the caller SAID, from streaming ASR, as a STOP / KEEP cue.

Acoustics cannot tell a one-word command ("توقف", "لا") from a one-word
acknowledgement ("نعم", "أها"): same length, same level. The words can.

    lexicon   classify(text) -> "stop" | "keep" | "content" | None
    worker    WordsWorker: runs an ASR backend off the audio thread; results
              carry the request id so stale ones (old run/utterance) are dropped
    backends  FasterWhisperBackend (offline CPU, MIT), or any object with
              transcribe(int16 audio @16 kHz) -> str

Nothing here blocks on_frame. In offline evaluation (`sync=True`) the backend
runs inline and its result is released only after `latency_ms` of stream
time, so replay scores include the ASR delay.
"""
from __future__ import annotations

import queue
import re
import threading
import time
import weakref
from typing import Optional

import numpy as np

# ------------------------------------------------------------------ lexicon
_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
_NON_LETTER = re.compile(r"[^ء-يa-z0-9 ]+")


def normalize(text: str) -> str:
    t = _DIACRITICS.sub("", text.lower())
    t = re.sub("[إأآٱ]", "ا", t)
    t = t.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    t = _NON_LETTER.sub(" ", t)
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)          # "اممممم" -> "امم"
    return " ".join(t.split())


# Words that claim the floor: objection, stop, wait, correction (Gulf/Najdi/Hijazi + MSA)
STOP_WORDS = {
    "لا", "لاء", "لالا", "توقف", "وقف", "قف", "اوقف", "لحظه", "لحظة", "دقيقه", "ثانيه", "استنى", "استني", "انتظر",
    "اصبر", "بس", "لاحظه", "اسمع", "اسمعني", "اسمعيني", "الغ", "الغي", "الغاء", "كنسل", "سوري", "عفوا", "مو", "مب", "ماابي",
    "مابي", "ماابغى", "مابغى", "غلط", "خطا", "ثواني", "يكفي", "خلاص", "stop", "no", "wait", "cancel",
}
# Listener tokens: acknowledge and hand the floor back
KEEP_WORDS = {
    "نعم", "ايه", "ايوه", "اي", "اها", "اه", "هيه", "هيه", "ام", "امم", "مم", "ممم", "تمام", "صح", "طيب", "موافق",
    "اوكي", "اوك", "زين", "حلو", "ماشي", "اكيد", "يب", "يس", "اهاا", "معك", "معاك", "معايا", "فاهم", "فاهمه",
    "مزبوط", "صحيح", "الله", "يعطيك", "العافيه", "تفضل", "تفضلي", "هلا", "ok", "yes", "yeah", "mhm",
}
# Multi-word stop phrases (checked on the normalised string)
STOP_PHRASES = ("ما ابي", "ما ابغى", "مو كذا", "مب كذا", "ثانيه بس", "خلني اقول", "بقول لك", "عندي سوال")


def _edit1(a: str, b: str) -> bool:
    """True if a and b differ by at most one insert/delete/substitute."""
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) > len(b):
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i + (len(a) == len(b)):] == b[i + 1:]


# only distinctive command words get fuzzy matching ("ماشي" must not become "مابي")
_FUZZY_STOP = ["لحظه", "توقف", "اوقف", "دقيقه", "استنى", "استني", "انتظر", "اسمع", "اسمعني", "الغاء", "ثانيه",
               "ثواني", "كنسل", "اصبر"]


def _is_stop(w: str) -> bool:
    # ASR slips on longer words ("لاحظه" for "لحظه"): allow one edit there;
    # short words (لا، بس، مو) must match exactly
    return w in STOP_WORDS or (len(w) >= 4 and w not in KEEP_WORDS and any(_edit1(w, s) for s in _FUZZY_STOP))


def classify(text: str) -> Optional[str]:
    """'stop' if any floor-claiming word, 'keep' if only listener tokens,
    'content' for other speech (>= 2 words), None if nothing usable."""
    t = normalize(text)
    if not t:
        return None
    if any(p in t for p in STOP_PHRASES):
        return "stop"
    toks = t.split()
    if any(_is_stop(w) for w in toks):
        return "stop"
    other = [w for w in toks if w not in KEEP_WORDS]
    if not other:
        return "keep"
    return "content" if len(other) >= 2 else None


# ------------------------------------------------------------------ backends
class FasterWhisperBackend:
    """Offline Whisper on CPU via CTranslate2 (pip install faster-whisper).
    model_dir: a local CTranslate2 Whisper model (e.g. faster-whisper-base)."""

    def __init__(self, model_dir: str, threads: int = 2):
        from faster_whisper import WhisperModel  # optional dependency
        self.model = WhisperModel(model_dir, device="cpu", compute_type="int8", cpu_threads=threads)
        self.transcribe(np.zeros(16000, dtype=np.int16))  # warm-up before the call starts

    def transcribe(self, audio: np.ndarray, **_ctx) -> str:
        x = audio.astype(np.float32) / 32768.0
        # no initial_prompt: Whisper copies prompt words into the output on
        # short clips (a "stop" word out of thin air). Short output cap: the
        # cues are in the first few words, and it stops repetition loops.
        segs, _ = self.model.transcribe(
            x, language="ar", beam_size=1, best_of=1, temperature=0.0, condition_on_previous_text=False,
            without_timestamps=True, vad_filter=False, no_speech_threshold=0.6, max_new_tokens=12)
        return " ".join(s.text for s in segs if s.no_speech_prob < 0.6).strip()


class WhisperKWSBackend:
    """Whisper tiny used twice on ONE encoding of the clip (~190 ms on 2 CPU threads):

    1. free transcript (greedy, <= 12 tokens), classified by the lexicon;
    2. keyword spotting: the probability Whisper gives each candidate word
       (forced alignment). One isolated word is where free transcription
       fails ("توقف" -> "تبقص"), but "is it توقف or نعم?" is easy for it.

    Speed: the clip is encoded in a 10 s window instead of Whisper's 30 s
    (encode ~40 ms instead of ~350 ms), with a small candidate list.
    If keyword spotting is confident that a stop word was said, that word is
    put in front of the transcript, so the lexicon sees it.
    Thresholds were chosen on dev + xdev (18 STOP / 32 KEEP events):
    12/18 caught (free transcript alone: 8/18), 0/32 false, ~190 ms per request.
    """

    STOP_CANDIDATES = ("لا", "توقف", "لحظة", "إلغاء", "اسمع")
    # common listener words MUST be candidates: a word missing from the list is
    # matched to its nearest candidate ("طيب" -> "توقف" before "طيب" was added)
    KEEP_CANDIDATES = ("نعم", "إيه", "أيوه", "أها", "طيب", "تمام")

    def __init__(self, model_dir: str, threads: int = 2, window_frames: int = 1000,
                 margin: float = 0.3, min_score: float = -2.5, no_speech: float = 0.6):
        import ctranslate2  # optional dependency (comes with faster-whisper)
        from faster_whisper import WhisperModel
        from faster_whisper.tokenizer import Tokenizer
        self._ct2 = ctranslate2
        self.model = WhisperModel(model_dir, device="cpu", compute_type="int8", cpu_threads=threads)
        self.tok = Tokenizer(self.model.hf_tokenizer, True, task="transcribe", language="ar")
        self.start = list(self.tok.sot_sequence) + [self.tok.no_timestamps]
        self.cands = list(self.STOP_CANDIDATES) + list(self.KEEP_CANDIDATES)
        self.cand_tokens = [self.tok.encode(" " + c) for c in self.cands]
        self.n_stop = len(self.STOP_CANDIDATES)
        self.window, self.margin, self.min_score, self.no_speech = window_frames, margin, min_score, no_speech
        self.last_scores: dict = {}
        self.transcribe(np.zeros(16000, dtype=np.int16))  # warm-up before the call starts

    def transcribe(self, audio: np.ndarray, **_ctx) -> str:
        x = audio.astype(np.float32) / 32768.0
        f = self.model.feature_extractor(x)[:, -self.window:]       # newest audio if longer than the window
        n = f.shape[-1]
        f = np.pad(f, ((0, 0), (0, self.window - n)))
        enc = np.array(self.model.encode(f))
        ct2 = self._ct2
        g = self.model.model.generate(ct2.StorageView.from_array(enc), [self.start], beam_size=1,
                                      max_length=len(self.start) + 12, return_no_speech_prob=True)[0]
        if g.no_speech_prob > self.no_speech:
            self.last_scores = {}
            return ""
        text = self.tok.decode([t for t in g.sequences_ids[0] if t < self.tok.eot]).strip()
        res = self.model.model.align(ct2.StorageView.from_array(np.repeat(enc, len(self.cands), axis=0)),
                                     self.start, self.cand_tokens, n)
        sc = [float(np.mean(np.log(np.asarray(r.text_token_probs[:len(t)]) + 1e-9)))
              for r, t in zip(res, self.cand_tokens)]
        self.last_scores = dict(zip(self.cands, sc))
        i_stop = int(np.argmax(sc[:self.n_stop]))
        best_stop, best_keep = sc[i_stop], max(sc[self.n_stop:])
        if best_stop - best_keep >= self.margin and best_stop >= self.min_score:
            text = f"{self.cands[i_stop]} {text}".strip()      # keyword spotting heard a stop word
        return text


# ------------------------------------------------------------------ worker
_BACKENDS: dict = {}
_BACKENDS_LOCK = threading.Lock()


def shared_backend(model_dir: str, threads: int = 2, kws: bool = True):
    """One loaded model per process, shared by every MySystem instance (a
    server that creates a system per call must not reload Whisper each time)."""
    key = (model_dir, threads, kws)
    with _BACKENDS_LOCK:
        if key not in _BACKENDS:
            cls = WhisperKWSBackend if kws else FasterWhisperBackend
            _BACKENDS[key] = cls(model_dir, threads=threads)
        return _BACKENDS[key]


def _serve(q: "queue.Queue", backend, ready: list, lock: threading.Lock, owner: "weakref.ref"):
    """Worker thread. Holds no reference to the WordsWorker itself, so the
    thread ends shortly after its owner (one call / one demo session) is gone."""
    while owner() is not None:
        try:
            req_id, audio = q.get(timeout=2.0)
        except queue.Empty:
            continue
        try:
            text = backend.transcribe(audio)
        except Exception:
            text = ""
        with lock:
            ready.append((0.0, req_id, text))


class WordsWorker:
    """One request in flight at a time; newest result wins.

    submit(req_id, audio, now_ms) -> bool (False if busy)
    poll(now_ms) -> (req_id, text) or None
    """

    def __init__(self, backend, sync: bool = False, latency_ms: Optional[float] = None):
        self.backend = backend
        self.sync = sync
        self.latency_ms = latency_ms      # sync mode: fixed simulated delay (None = measured compute time)
        self._busy = False
        self._ready = []                  # (release_ms, req_id, text)
        self._lock = threading.Lock()
        if not sync:
            self._q: "queue.Queue" = queue.Queue(maxsize=1)
            threading.Thread(target=_serve, args=(self._q, backend, self._ready, self._lock, weakref.ref(self)),
                             daemon=True).start()

    @property
    def busy(self) -> bool:
        return self._busy

    def submit(self, req_id, audio: np.ndarray, now_ms: float, **ctx) -> bool:
        if self._busy:
            return False
        self._busy = True
        if self.sync:
            t0 = time.perf_counter()
            try:
                text = self.backend.transcribe(audio, now_ms=now_ms, **ctx)
            except Exception:
                text = ""
            dt = (time.perf_counter() - t0) * 1000 if self.latency_ms is None else self.latency_ms
            self._ready.append((now_ms + dt, req_id, text))
        else:
            self._q.put((req_id, audio))
        return True

    def poll(self, now_ms: float):
        with self._lock:
            due = [r for r in self._ready if r[0] <= now_ms]
            if not due:
                return None
            self._ready[:] = [r for r in self._ready if r[0] > now_ms]
        self._busy = False
        _, req_id, text = due[-1]
        return req_id, text
