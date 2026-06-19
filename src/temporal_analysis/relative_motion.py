import numpy as np


class RelativeMotionAnalyzer:
    """
    Berechnet Handgeschwindigkeit relativ zum Gesichtszentrum.
    
    Nutzt Kalman-gefilterte Positionen aus TemporalTracker und
    rohe Keypoints für die Gesichtsbreite (Ohr-zu-Ohr).
    
    Ausgabe: [0, 1] — invariant gegen Kamerabewegung, Distanz zur Kamera
    und Auflösung.
    """

    KP_LEFT_EAR = 3
    KP_RIGHT_EAR = 4

    def __init__(self):
        self.prev_left = None
        self.prev_right = None

    def compute(self, motion_state, keypoints):
        face_center = motion_state["face"]["position"]
        face_width = self._face_width(keypoints)
        if face_width == 0:
            face_width = 1.0

        left_wrist = motion_state["left_wrist"]["position"]
        right_wrist = motion_state["right_wrist"]["position"]

        rel_left = (left_wrist - face_center) / face_width
        rel_right = (right_wrist - face_center) / face_width

        speed_left = (
            float(np.linalg.norm(rel_left - self.prev_left))
            if self.prev_left is not None else 0.0
        )
        speed_right = (
            float(np.linalg.norm(rel_right - self.prev_right))
            if self.prev_right is not None else 0.0
        )

        self.prev_left = rel_left
        self.prev_right = rel_right

        return min(max(speed_left, speed_right), 1.0)

    def reset(self):
        self.prev_left = None
        self.prev_right = None

    def _face_width(self, keypoints):
        if keypoints is None or len(keypoints) == 0:
            return 0.0
        person = keypoints[0]
        if len(person) <= max(self.KP_LEFT_EAR, self.KP_RIGHT_EAR):
            return 0.0
        left_ear = person[self.KP_LEFT_EAR]
        right_ear = person[self.KP_RIGHT_EAR]
        if float(left_ear[2]) < 0.5 or float(right_ear[2]) < 0.5:
            return 0.0
        return abs(float(right_ear[0]) - float(left_ear[0]))
