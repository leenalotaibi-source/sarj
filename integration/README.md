# Add SARJ v3 to the demo site (rayancis/sarj-turn-taking)

`demo-add-sarj-v3.patch` is one commit on top of `feat/demo-ui` @ `3b76ebb`. It adds SARJ v3 as a third detector
next to Baseline and P3. `demo-preview-BC-D02.png` shows the page after the patch: BC-D02, where Baseline and P3 stop
on «أها» and SARJ v3 keeps talking.

## Apply (repository owner)

```bash
git fetch origin feat/demo-ui && git checkout feat/demo-ui
git am path/to/demo-add-sarj-v3.patch       # one commit, applies cleanly on 3b76ebb
python -m pytest -q tests/                  # 146 passed when prepared
git push origin feat/demo-ui                # render.yaml deploys this branch automatically
```

Before pushing, run it locally:

```bash
pip install faster-whisper==1.2.1
python -c "from huggingface_hub import snapshot_download; snapshot_download('Systran/faster-whisper-tiny', local_dir='demo/models/sarj_v3/models/faster-whisper-tiny')"
python demo/server.py                       # http://127.0.0.1:8770
```

## What the patch changes

| File | Change |
|---|---|
| `demo/models/sarj_v3/` | SARJ v3 code (`my_system.py` + `tt/`), identical to leenalotaibi-source/sarj @ `18471a8` |
| `demo/server.py` | third detector `v3`, pinned **file by file** (`my_system.py` and every `tt/*.py`); a changed module disables it |
| `demo/static/*` | the page shows any number of detectors (cards, verdict line, results table, timeline lanes, benchmark); v3 is optional, Baseline and P3 stay required |
| `demo/benchmark.json`, `experiments/evidence/` | official real-time dev run of the exact served files: **Core 96.38**, R 0.951, S 0.969, 1 false stop, 0 misses, 0 protocol errors (`v3-18471a8-rt`) |
| `demo/Dockerfile`, `demo/requirements-public.txt` | `faster-whisper==1.2.1`; Whisper tiny (MIT) downloaded at image build. Without it v3 runs on acoustics alone and its description says so |
| `tests/` | 3 detectors, v3 refuses a changed module, 3 benchmark rows equal to `runs.json` |

## Checked / not checked

- Checked:
  - The full test suite passes (146).
  - Public mode was started from exactly the files the Dockerfile copies, in a clean environment, with `OMP_NUM_THREADS=1`.
  - Memory: 333 MB idle, 408 MB peak with 3 simultaneous playbacks. `SARJ_DEMO_MAX_SESSIONS` stays at 12 (4 playbacks).
- Not checked:
  - The Docker image itself (no Docker daemon was available).
  - Speed on Render's free CPU. If speech recognition is slow there, SARJ v3's words arrive late. Very late stop words are ignored, so it behaves like its acoustic part (dev Core 92.4) rather than doing anything wrong.
