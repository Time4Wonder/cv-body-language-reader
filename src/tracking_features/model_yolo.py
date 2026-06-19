import cv2
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
