# Real-Time Body Language Reader v1.0

Ein Computer-Vision-Projekt zur echtzeitfähigen Analyse von Mimik und Gestik. Entwickelt im Rahmen einer Projektarbeit an der Technischen Hochschule Deggendorf (THD).

[![Demo Video](https://img.youtube.com/vi/Emt20UvHRZ4/0.jpg)](https://youtu.be/Emt20UvHRZ4)

## Features

- **Echtzeit-Webcam-Pipeline** – YOLO Pose + ResNet Emotionen + Kalman Tracking in Echtzeit
- **Pose Estimation (YOLO11)** – Erfasst Körperhaltung und Handpositionen über 17 COCO-Keypoints
- **Gesichtsextraktion** – Automatisches Croppen des Gesichts aus den YOLO-Keypoints
- **Emotionserkennung (ResNet18)** – Klassifiziert 7 Emotionen trainiert auf FER-2013 (angry, disgust, fear, happy, neutral, sad, surprise)
- **Kalman-Filter-Tracking** – Rauschunterdrückte Verfolgung von Hand- und Gesichtspositionen
- **Relative Handgeschwindigkeit** – Geschwindigkeit relativ zum Gesichtszentrum (kamerainvariant, normalisiert auf [0,1])
- **Behavior Interpretation** – 7×3-Matrix (Emotion × Geschwindigkeits-Level → 21 deutsche Verhaltenslabels)
- **5-Sekunden-Rolling-Average** – Geglättete Verhaltensinterpretation
- **Live Chart** – Echtzeit-Matplotlib-Diagramm mit Emotionsverläufen
- **Session Report** – Zusammenfassung nach Beenden (dominante Emotion, Bewegungsstatistiken, Balkendiagramm)

## Voraussetzungen

- Python 3.10+
- Webcam
- Ubuntu/Linux (getestet)

## Installation

```bash
# Repository klonen
git clone https://github.com/Time4Wonder/cv-body-language-reader.git
cd cv-body-language-reader

# Virtuelle Umgebung erstellen und aktivieren
python3 -m venv .venv
source .venv/bin/activate

# PyTorch installieren
pip install torch torchvision

# Weitere Abhängigkeiten installieren
pip install -r requirements.txt
```

Das YOLO-Modell (`yolo11n-pose.pt`) wird automatisch beim ersten Start heruntergeladen.

## Verwendung

```bash
source .venv/bin/activate
python src/main.py
```

**Steuerung:**
- `q` – Beenden + Session-Report anzeigen

## Modell-Performance

Trainiert auf **FER-2013** (7 Klassen, 35.887 Graustufenbilder, 48×48 Pixel):

| Modell | Accuracy |
|--------|----------|
| resnet_fer2013v3.pth | **69%** |
| best_resnet_ferv4-chkpt.pth | **68%** |
| resnet_fer2013v2.pth | 67% |
| best_resnet_ferv5-freeze-chckpt.pth | 64% |

Alle trainierten Modelle liegen unter `models/`.

## Tech Stack

- **Framework:** PyTorch, TorchVision
- **Computer Vision:** OpenCV, Ultralytics YOLO
- **Visualisierung:** Matplotlib
- **Tracking:** Kalman Filter
- **Sprache:** Python 3.12

## Projektstruktur

```
cv-body-language-reader/
├── src/
│   ├── main.py                       # Einstiegspunkt
│   ├── tracking_features/
│   │   ├── model_yolo.py             # YOLO Pose Estimator
│   │   └── face_processor.py         # Gesichtsextraktion aus Keypoints
│   ├── spatial_analysis/
│   │   ├── dataset.py                # FER-2013 DataLoader
│   │   ├── model_resnet.py           # ResNet18 Expression Analyzer
│   │   └── train.py                  # Training-Script
│   ├── temporal_analysis/
│   │   ├── temporal_tracker.py       # Kalman-Filter-Tracking
│   │   ├── temporal_aggregator.py    # Zeitfenster-Aggregation
│   │   └── relative_motion.py        # Relative Handgeschwindigkeit
│   └── output/
│       ├── behavior_interpreter.py   # Verhaltensinterpretation (7×3-Matrix)
│       └── live_chart.py             # Echtzeit-Matplotlib-Chart
├── assets/
│   └── body-ln-reader_demo.mov       # Demo-Video (lokal)
├── models/                           # Trainierte PyTorch-Gewichte
├── data/raw/                         # FER-2013 Datensatz
├── docs/                             # Projektdokumentation
├── notebooks/                        # Jupyter-Notebooks
├── requirements.txt                  # Abhängigkeiten
└── README.md
```

## Autoren

- **Mohammad Samali**
- **Emmanuel Da'si Franck**
- **Lorenz Burgard**
