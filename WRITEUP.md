# Write-up (draft)

## External models, libraries and data (must stay complete until code freeze)

| Item | Used for | Runtime? | Licence |
|---|---|---|---|
| numpy | everything | yes | BSD-3 |
| onnxruntime | Silero VAD inference (optional) | optional | MIT |
| Silero VAD v5 (`models/silero_vad.onnx`) | speech/non-speech per frame (off by default) | optional | MIT |
| scipy, soundfile | synthetic data tools | no | BSD-3 / BSD-3 |
| scikit-learn | classifier training | no | BSD-3 |

No internet access is needed at runtime.

## Failure cases (≥ 5, with Arabic examples)
_To fill from `python -m ttbench.summary runs/<name>`._
