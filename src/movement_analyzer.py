import numpy as np

class MovementAnalyzer:
    """
    Berechnet die Bewegungsintensität des Körpers zwischen zwei Frames.
    Ausgabe: Wert zwischen 0 und 1
    0 = keine Bewegung
    1 = starke Bewegung
    """

    def __init__(self, max_movement=50, smoothing=0.8):
        self.previous_keypoints = None
        self.max_movement = max_movement
        self.smoothing = smoothing
        self.smoothed_intensity = 0.0

    def calculate_movement_intensity(self, keypoints):
        """
        Berechnet die Bewegungsintensität anhand der YOLO-Keypoints.
        """
        if keypoints is None or len(keypoints) == 0:
            return 0.0

        # Erste erkannte Person nehmen
        current = keypoints[0].cpu().numpy()

        # Nur x, y Koordinaten benutzen
        current_xy = current[:, :2]

        # Falls es noch keinen vorherigen Frame gibt
        if self.previous_keypoints is None:
            self.previous_keypoints = current_xy
            return 0.0

        previous_xy = self.previous_keypoints

        # Distanz zwischen aktuellem und vorherigem Keypoint berechnen
        distances = np.linalg.norm(current_xy - previous_xy, axis=1)

        # Durchschnittliche Bewegung
        avg_movement = np.mean(distances)

        # Normalisierung auf 0 bis 1
        intensity = avg_movement / self.max_movement
        intensity = np.clip(intensity, 0.0, 1.0)

        # Glättung, damit Wert nicht stark springt
        self.smoothed_intensity = (
            self.smoothing * self.smoothed_intensity
            + (1 - self.smoothing) * intensity
        )

        # Aktuelle Keypoints für nächsten Frame speichern
        self.previous_keypoints = current_xy

        return float(self.smoothed_intensity)