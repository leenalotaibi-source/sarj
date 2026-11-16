#!/usr/bin/env bash
# Plumbing test without the kit or real data: fake clips -> synth calls -> replay -> train.
set -euo pipefail
cd "$(dirname "$0")/.."
T=$(mktemp -d)
python3 tests/fake_clips.py "$T/clips"
python3 tools/make_synth_calls.py --clips "$T/clips" --out "$T/train" --n 40 --seed 1
python3 tools/make_synth_calls.py --clips "$T/clips" --out "$T/test" --n 30 --seed 7
python3 tools/replay.py "$T/test" --fails 3
python3 tools/train_classifier.py "$T/train" --out "$T/clf.json"
TT_CFG="use_clf=1,clf_path=$T/clf.json" python3 tools/replay.py "$T/test" --fails 0 | tail -1
rm -rf "$T"
