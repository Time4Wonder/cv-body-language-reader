import numpy as np
import cv2


class KalmanPoint2D:
    """
    Kalman-Filter für einen 2D-Punkt mit konstantem Geschwindigkeitsmodell.
    """

    def __init__(self, dt=1.0, process_noise=1.0, measurement_noise=10.0):
        """
        Initialisiere den Kalman-Filter.
        """
        # Zustand und Kovarianz
        self.x = np.zeros((4, 1), dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 1000.0

        # Transitionsmodell
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=np.float32)

        # Messmodell
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        # Rauschmatrizen
        self.Q = np.eye(4, dtype=np.float32) * process_noise
        self.R = np.eye(2, dtype=np.float32) * measurement_noise
        self.I = np.eye(4, dtype=np.float32)

        self.initialized = False

    def predict(self):
        """
        Vorhersage des nächsten Zustands.
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].flatten()  # vorhergesagte Position (px, py)

    def update(self, measurement):
        """
        Aktualisiere den Zustand mit einer neuen Messung.
        """
        z = np.array(measurement, dtype=np.float32).reshape(2, 1)

        # Initialisierung beim ersten Messwert
        if not self.initialized:
            self.x[0], self.x[1] = z[0], z[1]
            self.x[2], self.x[3] = 0.0, 0.0
            self.initialized = True
            return self.x[:2].flatten()

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (self.I - K @ self.H) @ self.P
        return self.x[:2].flatten()

    def position(self):
        """Aktuelle geschätzte Position (px, py)."""
        return self.x[:2].flatten()

    def velocity(self):
        """Aktuelle geschätzte Geschwindigkeit (vx, vy) in Pixeln/Frame."""
        return self.x[2:].flatten()

    def speed(self):
        """Betrag der Geschwindigkeit."""
        vx, vy = self.velocity()
        return float(np.hypot(vx, vy))


class TemporalTracker:
    """
    Verwaltet mehrere Kalman-Filter für die zeitliche Verfolgung.
    """

    # COCO-Pose Keypoint-Indizes (wie in face_processor.py verwendet)
    KP_NOSE = 0
    KP_LEFT_EAR = 3
    KP_RIGHT_EAR = 4
    KP_LEFT_WRIST = 9
    KP_RIGHT_WRIST = 10

    def __init__(self, conf_threshold=0.5, max_misses=15,
                 process_noise=1.0, measurement_noise=10.0):
        """
        Initialisiere den Temporal Tracker.
        """
        self.conf_threshold = conf_threshold
        self.max_misses = max_misses

        # Ein Kalman-Filter pro verfolgtem Punkt
        self.filters = {
            "left_wrist":  KalmanPoint2D(process_noise=process_noise,
                                         measurement_noise=measurement_noise),
            "right_wrist": KalmanPoint2D(process_noise=process_noise,
                                         measurement_noise=measurement_noise),
            "face":        KalmanPoint2D(process_noise=process_noise,
                                         measurement_noise=measurement_noise),
        }
        # Zaehler fuer aufeinanderfolgende Frames ohne gueltige Messung
        self.misses = {name: 0 for name in self.filters}

    def _valid(self, point):
        """Prüfe ob ein Keypoint zuverlässig ist."""
        if point is None:
            return False
        if len(point) >= 3 and float(point[2]) < self.conf_threshold:
            return False
        if float(point[0]) == 0.0 and float(point[1]) == 0.0:
            return False
        return True

    def _face_center(self, person_kpts):
        """
        Berechne das Gesichtszentrum aus Nase und Ohren.
        """
        candidates = []
        for idx in (self.KP_NOSE, self.KP_LEFT_EAR, self.KP_RIGHT_EAR):
            if idx < len(person_kpts) and self._valid(person_kpts[idx]):
                candidates.append(person_kpts[idx][:2])
        if not candidates:
            return None
        pts = np.array(candidates, dtype=np.float32)
        return pts.mean(axis=0)

    def update(self, keypoints):
        """
        Update mit neuen Keypoints.
        """
        measurements = {"left_wrist": None, "right_wrist": None, "face": None}

        if keypoints is not None and len(keypoints) > 0:
            person = keypoints[0]
            if hasattr(person, "cpu"):
                person = person.cpu().numpy()

            if len(person) > self.KP_LEFT_WRIST and self._valid(person[self.KP_LEFT_WRIST]):
                measurements["left_wrist"] = person[self.KP_LEFT_WRIST][:2]
            if len(person) > self.KP_RIGHT_WRIST and self._valid(person[self.KP_RIGHT_WRIST]):
                measurements["right_wrist"] = person[self.KP_RIGHT_WRIST][:2]
            measurements["face"] = self._face_center(person)

        output = {}
        for name, kf in self.filters.items():
            kf.predict()

            meas = measurements[name]
            if meas is not None:
                kf.update(meas)
                self.misses[name] = 0
                visible = True
            else:
                self.misses[name] += 1
                visible = self.misses[name] <= self.max_misses

            output[name] = {
                "position": kf.position(),
                "velocity": kf.velocity(),
                "speed": kf.speed(),
                "visible": visible and kf.initialized,
            }
        return output

    def draw(self, frame, state=None):
        """
        Zeichne die verfolgten Punkte und Geschwindigkeiten in das Bild.
        """
        if state is None:
            state = {
                name: {
                    "position": kf.position(),
                    "velocity": kf.velocity(),
                    "speed": kf.speed(),
                    "visible": kf.initialized,
                }
                for name, kf in self.filters.items()
            }

        colors = {
            "left_wrist":  (0, 255, 0),    # gruen
            "right_wrist": (0, 165, 255),  # orange
            "face":        (255, 0, 0),    # blau
        }

        for name, s in state.items():
            if not s["visible"]:
                continue
            px, py = int(s["position"][0]), int(s["position"][1])
            vx, vy = s["velocity"]
            color = colors.get(name, (255, 255, 255))

            cv2.circle(frame, (px, py), 6, color, -1)
            cv2.arrowedLine(frame, (px, py),
                            (int(px + vx * 5), int(py + vy * 5)),
                            color, 2, tipLength=0.3)
            cv2.putText(frame, f"{name}: {s['speed']:.1f} px/f",
                        (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, cv2.LINE_AA)

        return frame
