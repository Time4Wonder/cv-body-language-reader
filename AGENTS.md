# AGENTS.md — Real-Time Body Language Reader

German-documented THD projectarbeit. Live webcam pipeline fusing YOLO pose + ResNet emotions + temporal analysis.

## Merged into main ✓

| Branch | What it added |
|--------|-------------|
| `feature/resnet-mimik` | ResNet18 emotion classifier (FER-2013, 7 classes), dataset loader, training loop |
| `feature/time_analysis` | Kalman wrist/face tracking, temporal frame aggregation, session reports |

Still in `origin/head-velocity-tracking` (not merged):
MovementAnalyzer, GestureAnalyzer, FeatureFusion (import bug — see below)

## Architecture

`src/main.py` is the single entrypoint. Subdirectories:
- `src/spatial_analysis/` — ResNet emotion model, dataset loader, training
- `src/temporal_analysis/` — Kalman tracking, temporal aggregator, RelativeMotionAnalyzer
- `src/tracking_features/` — YOLO pose estimator, face extraction
- `src/output/` — LiveChart matplotlib visualization, BehaviorInterpreter

Pipeline:
`Webcam → YOLO pose → face crop → ResNet emotions + Kalman tracking → RelativeMotionAnalyzer → temporal aggregation + behavior interpretation → session report`

BehaviorInterpreter: 7×3 Matrix (Emotion × Speed-Level → deutsches Verhaltenslabel). Eigenes OpenCV-Fenster `"Behavior Interpretation"` mit Label, Emotion, Speed-Balken, Alternativen.

RelativeMotionAnalyzer: Handgeschwindigkeit relativ zum Gesichtszentrum (Kopfbreite als Einheit) → [0,1], invariant gegen Kamerabewegung und Distanz.

Keypoints (COCO): nose=0, left_ear=3, right_ear=4, left_wrist=9, right_wrist=10

## Gotchas

- **requirements.txt incomplete** — declares only `ultralytics` + `opencv-python`. ResNet branch uses `torch` + `torchvision` (installed in `.venv` but undeclared).
- **EMOTIONEN order mismatch** — `ImageFolder` sorts folder names alphabetically (`angry=0, ..., neutral=4, ..., surprise=6`). FER-2013 training used a different mapping. Fix the `EMOTIONEN` list in `spatial_analysis/model_resnet.py` after merge.
- **head-velocity-tracking import bug** — `main.py` does `from model_yolo import MovementAnalyzer, GestureAnalyzer` but those live in separate files.
- **No test/lint/typecheck** — only `pyright --basic`. No CI, no pre-commit.
- **Activate `.venv/` first** — system python lacks torch.

## Commands

```bash
source .venv/bin/activate
python src/main.py              # launch webcam pipeline
python src/spatial_analysis/train.py  # train ResNet on FER-2013 (from resnet-mimik)
```

Training data layout: `data/raw/{train,test}/{class_folder}/`
