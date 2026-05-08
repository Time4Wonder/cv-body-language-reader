# Real-Time Body Language Reader

Ein Computer-Vision-Projekt zur echtzeitfähigen Auslesung und Prognose von Mimik und Gestik. Entwickelt im Rahmen einer Projektarbeit an der Technischen Hochschule Deggendorf (THD).

## Projektbeschreibung
Dieses Projekt zielt darauf ab, menschliche Körpersprache nicht nur starr in Basisemotionen zu klassifizieren, sondern durch eine tiefere Analyse von Mimik und Gestik komplexe psychologische Zustände und fließende Emotionen abzubilden. 

Das System nutzt einen **Late-Fusion-Ansatz**:
1. **Mimik-Analyse (CNN):** Ein Convolutional Neural Network analysiert Gesichtsmerkmale und zielt auf die Vorhersage von Soft-Labels ab (Compound Expressions), um Nuancen in der Mimik besser darzustellen.
2. **Gestik- & Haltungsanalyse (YOLO):** Ein YOLO-basiertes Modell erfasst parallel die Körperhaltung und Handgesten.
3. **Late-Fusion Classification:** Die Outputs beider Netzwerke werden in den späten Schichten zusammengeführt, um eine ganzheitliche Prognose des emotionalen und gestischen Zustands zu treffen.

## Tech Stack
* **Framework:** PyTorch
* **Computer Vision:** OpenCV
* **Architektur:** CNN (Custom / Pre-trained), YOLO (Ultralytics)
* **Datenverarbeitung:** Pandas, NumPy
* **Environment:** Python 3.x, lokal lauffähig auf Ubuntu/Linux

## Datensätze
Das Training und die Validierung des Modells sind modular aufgebaut, um verschiedene Datensätze verarbeiten zu können:
* **FER-2013:** Initialer Proof of Concept (PoC) zum Aufbau der PyTorch-Pipeline und für schnelle lokale Tests.
* **AffectNet+ / Aff-Wild2:** Ziel-Datensätze für das finale Training, um Valence/Arousal, Action Units und Soft-Labels für komplexe Emotionen in-the-wild auswerten zu können (Zugang via Deep Learning Lab).


## Projektstruktur

```text
body-language-reader/
│
├── data/                  # Trainings- und Testdaten (FER-2013, etc.)
│   ├── raw/               # Unveränderte Originaldaten
│   └── processed/         # Bereinigte und transformierte Tensoren
│
├── models/                # Gespeicherte PyTorch-Weights (.pth)
│
├── src/                   # Source Code
│   ├── dataset.py         # PyTorch Dataset und DataLoader Logik
│   ├── model_cnn.py       # CNN Architektur für die Mimik
│   ├── model_yolo.py      # YOLO-Integration für die Gestik
│   ├── fusion.py          # Late-Fusion Logik
│   ├── train.py           # Manueller PyTorch Training-Loop
│   └── evaluate.py        # Metriken und Validierung
│
├── notebooks/             # Jupyter/NotebookLM Skizzen und Analysen
├── requirements.txt       # Python Dependencies
└── README.md

```

## Autoren
* **Mohammad Samali**
* **Ed Franck**
* **Lorenz Burgard**

