import cv2
import numpy as np
from ultralytics import YOLO

class PoseEstimator:
    """
    Diese Klasse kapselt das YOLO-Pose-Modell. 
    Sie ist zuständig für die Erkennung von Skeletten und Keypoints im Bild.
    """
    def __init__(self, model_path: str = 'yolo11n-pose.pt'):
        """
        Initialisiert das YOLO-Modell.
        :param model_path: Pfad zur Model-Datei (.pt)
        """
        self.model = YOLO(model_path)
        self.cap = None
    
    def estimate(self, frame):
        """
        Führt die Pose-Schätzung auf einem einzelnen Bild aus.
        :param frame: Das aktuelle Kamerabild (OpenCV BGR Format)
        :return: Ein Generator-Objekt mit den Ergebnissen der Erkennung.
        """
        return self.model(frame, verbose=False, stream=True)
    
    def extract_keypoints(self, results):
        """
        Extrahiert die rohen Keypoint-Daten aus den YOLO-Ergebnissen.
        :param results: Die Ergebnisse der estimate-Funktion.
        :return: Ein Tensor/Array mit den Keypoints [Personen, Punkte, [x,y,conf]] oder None.
        """
        for r in results:
            if r.keypoints is not None:
                return r.keypoints.data
        return None

    def draw(self, results):
        """
        Zeichnet das standardmäßige YOLO-Skelett und die Boxen in das Bild.
        :param results: Die Ergebnisse der estimate-Funktion.
        :return: Ein neues Bild mit den eingezeichneten Skeletten oder None.
        """
        for r in results:
            return r.plot()
        return None

class MovementAnalyzer:
    """
    Berechnet die Bewegungsintensität des Körpers anhand der YOLO-Keypoints.
    Der Ausgabewert liegt zwischen 0 und 1.
    """

    def __init__(self, max_movement=80, smoothing=0.8):
        # Speichert die Keypoints des vorherigen Frames
        self.previous_keypoints = None

        # Maximal erwartete Bewegung zur Normalisierung
        self.max_movement = max_movement

        # Glättungsfaktor zur Reduzierung starker Schwankungen
        self.smoothing = smoothing

        # Geglättete Bewegungsintensität
        self.smoothed_intensity = 0.0

    def calculate_movement_intensity(self, keypoints):
        """
        Berechnet die Bewegungsintensität zwischen aktuellem und vorherigem Frame.
        """

        if keypoints is None or len(keypoints) == 0:
            return 0.0

        # Erste erkannte Person verwenden
        current = keypoints[0].cpu().numpy()

        # Nur x- und y-Koordinaten verwenden
        current_xy = current[:, :2]

        # Beim ersten Frame gibt es noch keinen Vergleichswert
        if self.previous_keypoints is None:
            self.previous_keypoints = current_xy
            return 0.0

        previous_xy = self.previous_keypoints

        # Abstand jedes Keypoints zum vorherigen Frame berechnen
        distances = np.linalg.norm(current_xy - previous_xy, axis=1)

        # Durchschnittliche Bewegung aller Keypoints berechnen
        avg_movement = np.mean(distances)

        # Wert auf den Bereich 0 bis 1 normieren
        intensity = avg_movement / self.max_movement
        intensity = np.clip(intensity, 0.0, 1.0)

        # Wert glätten, damit die Anzeige nicht stark springt
        self.smoothed_intensity = (
            self.smoothing * self.smoothed_intensity
            + (1 - self.smoothing) * intensity
        )

        # Aktuelle Keypoints für den nächsten Frame speichern
        self.previous_keypoints = current_xy

        return float(self.smoothed_intensity)
    

class GestureAnalyzer:
    """
    Erkennt einfache Körpergesten anhand der YOLO-Pose-Keypoints.
    """

    def analyze(self, keypoints):
        """
        Gibt eine erkannte Geste als Text zurück.
        """

        if keypoints is None or len(keypoints) == 0:
            return "No Person"

        # Erste erkannte Person verwenden
        person = keypoints[0]

        # COCO-Keypoints:
        # 0 = Nase
        # 9 = linkes Handgelenk
        # 10 = rechtes Handgelenk
        nose = person[0]
        left_wrist = person[9]
        right_wrist = person[10]

        nose_y = nose[1]
        left_wrist_y = left_wrist[1]
        right_wrist_y = right_wrist[1]

        if left_wrist_y < nose_y and right_wrist_y < nose_y:
            return "Hands Up"

        if left_wrist_y < nose_y or right_wrist_y < nose_y:
            return "One Hand Up"

        return "Neutral"

    def encode(self, gesture):
        """
        Wandelt die erkannte Geste in numerische Werte für die Feature-Fusion um.
        Reihenfolge: [Hands Up, One Hand Up, Neutral]
        """

        if gesture == "Hands Up":
            return [1, 0, 0]

        if gesture == "One Hand Up":
            return [0, 1, 0]

        return [0, 0, 1]