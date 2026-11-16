"""Entry point for the Sarj turn-taking benchmark.

All logic lives in tt/ (pure, testable). This file only adapts it to the kit's
`System` interface and makes sure nothing ever raises out of a callback.

NOTE: written before we had the kit in the repo. Check against the kit's
example system (INTERFACE.md): how an action is emitted (return value vs a
send method) and the exact on_agent_start arguments. Adjust `_emit` only.
"""
from __future__ import annotations

import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tt.config import Config  # noqa: E402
from tt.decider import Decider  # noqa: E402

try:  # the kit provides the base class
    from ttbench.system import System  # type: ignore
except Exception:  # pragma: no cover - lets us run offline without the kit
    try:
        from ttbench import System  # type: ignore
    except Exception:
        class System:  # minimal stand-in
            def __init__(self, *a, **k):
                pass


def _config_from_env() -> Config:
    """TT_CFG="stop_after_ms=480,soft_stop=1" for scorer sweeps."""
    cfg = Config()
    spec = os.environ.get("TT_CFG", "").strip()
    for item in filter(None, (s.strip() for s in spec.split(","))):
        k, v = item.split("=", 1)
        cfg.override(**{k.strip(): v.strip()})
    return cfg


class MySystem(System):
    def __init__(self, *args, **kwargs):
        try:
            super().__init__(*args, **kwargs)
        except Exception:
            pass
        self.cfg = _config_from_env()
        self.dec = Decider(self.cfg)
        self.debug = bool(os.environ.get("TT_DEBUG"))
        self._slow = 0

    # ---------------------------------------------------------------- output
    def _emit(self, action):
        """Return the action; if the kit uses a send-style method, call it too."""
        if action is None:
            return None
        for name in ("send", "send_action", "emit"):
            fn = getattr(super(), name, None)
            if callable(fn):
                try:
                    fn(action)
                except Exception:
                    pass
                break
        return action

    # ------------------------------------------------------------- callbacks
    def on_agent_start(self, *args, **kwargs):
        try:
            self.dec.agent_start(*args, **kwargs)
        except Exception:
            self._err()

    def on_agent_end(self, *args, **kwargs):
        try:
            self.dec.agent_end(*args, **kwargs)
        except Exception:
            self._err()

    def on_frame(self, agent, caller):
        t0 = time.perf_counter()
        try:
            action = self.dec.frame(agent, caller)
        except Exception:
            self._err()
            action = None
        dt = time.perf_counter() - t0
        if dt > 0.010:
            self._slow += 1
            if self.debug:
                print(f"[tt] slow frame {dt*1000:.1f} ms", file=sys.stderr)
        if action and self.debug:
            print(f"[tt] {self.dec.log[-1]}", file=sys.stderr)
        return self._emit(action)

    def _err(self):
        if self.debug:
            traceback.print_exc()
