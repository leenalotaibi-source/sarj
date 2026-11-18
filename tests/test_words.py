"""Words layer: lexicon decisions and stale-result handling (no ASR model needed)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tt.config import Config  # noqa: E402
from tt.decider import Decider  # noqa: E402
from tt.words import WordsWorker, classify, normalize  # noqa: E402


def test_normalize_folds_spelling_variants():
    assert normalize("أَيْوَه") == normalize("ايوه")
    assert normalize("لحظة!") == "لحظه"
    assert normalize("اممممم") == "امم"


def test_classify():
    assert classify("لا لا … لحظة، بكرة أنا مو بالرياض.") == "stop"
    assert classify("توقف") == "stop"
    assert classify("ما أبي هذا") == "stop"
    assert classify("أها.") == "keep"
    assert classify("إيه إيه.") == "keep"
    assert classify("نعم، موافق.") == "keep"
    assert classify("الوطنية، أيوه، معايا.") is None   # one unknown word: no verdict
    assert classify("وين الشاحن يا ولد") == "content"
    assert classify("") is None


class _Fixed:
    def __init__(self, text):
        self.text = text

    def transcribe(self, audio, **_):
        return self.text


def test_sync_worker_releases_after_latency():
    w = WordsWorker(_Fixed("لا"), sync=True, latency_ms=300)
    assert w.submit("r", np.zeros(320, np.int16), now_ms=1000)
    assert not w.submit("r2", np.zeros(320, np.int16), now_ms=1020)   # one in flight
    assert w.poll(1200) is None
    assert w.poll(1300) == ("r", "لا")
    assert not w.busy


def test_stale_result_from_previous_utterance_is_ignored():
    d = Decider(Config(), words=WordsWorker(_Fixed("لا"), sync=True, latency_ms=0))
    d.agent_start("نص", 1000)
    old = (id(d), d.utt_seq - 1, d.run_seq)
    d.words._ready.append((0.0, old, "لا"))
    d.words._busy = True
    d.frame(np.zeros(320, np.int16), np.zeros(320, np.int16))
    assert d.word_verdict is None


def test_worker_thread_ends_with_its_owner():
    import gc
    import threading
    import time as _t
    before = threading.active_count()
    w = WordsWorker(_Fixed("نعم"))
    assert w.submit("r", np.zeros(320, np.int16), now_ms=0)
    for _ in range(100):
        if w.poll(0):
            break
        _t.sleep(0.01)
    assert threading.active_count() == before + 1
    del w
    gc.collect()
    _t.sleep(2.5)
    assert threading.active_count() == before


MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "faster-whisper-tiny")


def test_kws_backend_on_silence_and_candidates():
    import pytest
    if not os.path.isdir(MODEL):
        pytest.skip("Whisper tiny not downloaded (models/faster-whisper-tiny)")
    from tt.words import WhisperKWSBackend
    b = WhisperKWSBackend(MODEL)
    assert b.transcribe(np.zeros(16000, np.int16)) == ""          # no speech -> nothing, no keyword guess
    # every keep candidate is a listener token, every stop candidate a stop word, for the lexicon too
    assert all(classify(c) == "keep" for c in b.KEEP_CANDIDATES)
    assert all(classify(c) == "stop" for c in b.STOP_CANDIDATES)
    assert "طيب" in b.KEEP_CANDIDATES      # regression: without it, "طيب" was matched to "توقف"
