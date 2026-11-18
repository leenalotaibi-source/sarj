"""MySystem must start and run even when the optional ASR model is missing."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_starts_without_asr_model(monkeypatch):
    monkeypatch.setenv("TT_CFG", "use_words=1,asr_model_dir=/nonexistent/model")
    import my_system
    m = my_system.MySystem()
    assert m.words is None
    m.on_scenario_start({})
    m.on_agent_start("مرحبا، كيف أقدر أساعدك؟", 2000)
    for _ in range(50):
        assert m.on_frame(np.zeros(320, np.int16), np.zeros(320, np.int16)) is None
    m.on_agent_end()
