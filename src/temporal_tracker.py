import numpy as np
import cv2

# =============================================================================
#  Zeitliche Analyse (Image Sequence Processing)
# -----------------------------------------------------------------------------
#  Umsetzung des Kalman-Filters nach Vorlesung Computer Vision (Prof. Glauner),
#  Kapitel 7 "Image sequence processing" -> Abschnitt "Kalman filter".
#
#  Idee aus der Vorlesung (Folien 268-269):
#    "Filtering: estimating state variables (here, position and velocity)
#     from noisy observations over time. Our variables are continuous."
#  -> Genau unser Fall: YOLO liefert verrauschte, teils fehlende Positionen
#     (Handgelenke / Gesicht). Der Kalman-Filter glaettet die Trajektorie und
#     ueberbrueckt Aussetzer (Roadmap Phase 3).
#
#  Lineares Gauss'sches Modell (Folie 270):
#    "the next state X_{t+1} must be a linear function of the current state
#     X_t, plus some Gaussian noise."
# =============================================================================


class KalmanPoint2D:
    """
    Eigenstaendige (from-scratch) Implementierung eines Kalman-Filters fuer
    EINEN 2D-Punkt mit konstantem Geschwindigkeitsmodell (constant velocity).

    Zustandsvektor x (4x1):   [px, py, vx, vy]^T
        - Position (px, py) in Pixeln
        - Geschwindigkeit (vx, vy) in Pixeln pro Frame
    Messvektor z (2x1):       [px, py]^T  (nur Position wird von YOLO gemessen)

    Die Geschwindigkeit ist eine *versteckte* Zustandsvariable (X_t in der
    Vorlesung), die Position ist die *Beobachtung* (E_t). Vgl. Folie 237
    (States and observations) sowie das "bird's flight"-Beispiel (Folie 270),
    in dem Position UND Geschwindigkeit den Zustand bilden.
    """

    def __init__(self, dt=1.0, process_noise=1.0, measurement_noise=10.0):
        """
        :param dt: Zeitschritt zwischen zwei Frames (in Frames -> 1.0).
        :param process_noise: Prozessrauschen q. Groesser = Filter vertraut
               dem Modell weniger und reagiert schneller auf Aenderungen.
        :param measurement_noise: Messrauschen r. Groesser = Filter vertraut
               den (verrauschten) YOLO-Messungen weniger und glaettet staerker.
        """
        # --- Zustand x und Kovarianz P -------------------------------------
        # x_t: aktueller Schaetzwert des Zustands (Folie: belief state)
        self.x = np.zeros((4, 1), dtype=np.float32)
        # P: Unsicherheit ueber den Zustand. Start: hoch (wir wissen nichts).
        self.P = np.eye(4, dtype=np.float32) * 1000.0

        # --- Transitionsmodell F (Folie 240, "transition model") -----------
        # Konstantes Geschwindigkeitsmodell:
        #   px_{t+1} = px_t + vx_t * dt
        #   py_{t+1} = py_t + vy_t * dt
        #   vx, vy   = konstant (+ Rauschen)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=np.float32)

        # --- Sensormodell H (Folie 244, "sensor model" P(E_t | X_t)) -------
        # Wir messen nur die Position, nicht die Geschwindigkeit.
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        # --- Prozessrauschen Q (Gauss'sches Rauschen aus Folie 270) --------
        self.Q = np.eye(4, dtype=np.float32) * process_noise

        # --- Messrauschen R ------------------------------------------------
        self.R = np.eye(2, dtype=np.float32) * measurement_noise

        # Identitaetsmatrix fuer die Update-Gleichung
        self.I = np.eye(4, dtype=np.float32)

        self.initialized = False

    def predict(self):
        """
        VORHERSAGE-SCHRITT (FORWARD-Operator der Vorlesung, Folie 269/249).
        Projiziert den Zustand und die Unsicherheit einen Frame in die Zukunft.

            x_pred = F * x
            P_pred = F * P * F^T + Q

        WICHTIG fuer die Roadmap (Phase 3): Dieser Schritt laeuft auch dann,
        wenn YOLO KEINE Messung liefert (Hand kurz verloren). So bleibt eine
        plausible Position erhalten -> "Aussetzer abfangen".
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].flatten()  # vorhergesagte Position (px, py)

    def update(self, measurement):
        """
        KORREKTUR-SCHRITT (Filtering / state estimation, Folie 249).
        Korrigiert die Vorhersage mit einer neuen, verrauschten YOLO-Messung.

            y = z - H * x              (Innovation: Mess- minus Vorhersage)
            S = H * P * H^T + R        (Innovationskovarianz)
            K = P * H^T * S^-1         (Kalman-Gain)
            x = x + K * y
            P = (I - K * H) * P

        :param measurement: gemessene Position [px, py] von YOLO.
        """
        z = np.array(measurement, dtype=np.float32).reshape(2, 1)

        # Beim allerersten gueltigen Messwert: Zustand direkt darauf setzen,
        # damit der Filter nicht erst langsam von (0,0) "hereinfaehrt".
        if not self.initialized:
            self.x[0], self.x[1] = z[0], z[1]
            self.x[2], self.x[3] = 0.0, 0.0
            self.initialized = True
            return self.x[:2].flatten()

        y = z - self.H @ self.x                      # Innovation
        S = self.H @ self.P @ self.H.T + self.R      # Innovationskovarianz
        K = self.P @ self.H.T @ np.linalg.inv(S)     # Kalman-Gain

        self.x = self.x + K @ y                      # Zustand korrigieren
        self.P = (self.I - K @ self.H) @ self.P      # Kovarianz korrigieren
        return self.x[:2].flatten()

    def position(self):
        """Aktuelle geschaetzte Position (px, py)."""
        return self.x[:2].flatten()

    def velocity(self):
        """Aktuelle geschaetzte Geschwindigkeit (vx, vy) in Pixeln/Frame."""
        return self.x[2:].flatten()

    def speed(self):
        """Betrag der Geschwindigkeit (Skalar) in Pixeln/Frame."""
        vx, vy = self.velocity()
        return float(np.hypot(vx, vy))


class TemporalTracker:
    """
    Verwaltet mehrere KalmanPoint2D-Filter gleichzeitig und kapselt damit die
    komplette zeitliche Analyse fuer die Pipeline:

        - linkes Handgelenk  (COCO-Keypoint 9)
        - rechtes Handgelenk (COCO-Keypoint 10)
        - Gesichtszentrum    (Mittel aus Nase/Ohren bzw. Box-Mitte)

    Liefert geglaettete Positionen und Geschwindigkeiten -> direkte Grundlage
    fuer die Features aus Roadmap Phase 3 (z.B. durchschnittliche
    Handgeschwindigkeit ueber die letzten N Frames).
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
        :param conf_threshold: Mindest-Konfidenz, ab der ein Keypoint als
               gueltige Messung akzeptiert wird (YOLO liefert pro Punkt conf).
        :param max_misses: Wie viele Frames ein Punkt rein per Vorhersage
               ueberbrueckt wird, bevor er als "verloren" gilt.
        :param process_noise / measurement_noise: an alle Filter durchgereicht.
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

    # ------------------------------------------------------------------ #
    #  Hilfsfunktionen zur Extraktion der Messpunkte aus den Keypoints    #
    # ------------------------------------------------------------------ #
    def _valid(self, point):
        """Prueft, ob ein Keypoint [x, y, conf] zuverlaessig genug ist."""
        if point is None:
            return False
        # conf ist die dritte Komponente, falls vorhanden
        if len(point) >= 3 and float(point[2]) < self.conf_threshold:
            return False
        # YOLO setzt fehlende Punkte oft auf (0, 0)
        if float(point[0]) == 0.0 and float(point[1]) == 0.0:
            return False
        return True

    def _face_center(self, person_kpts):
        """
        Schaetzt das Gesichtszentrum aus Nase und Ohren (gemittelt ueber die
        verfuegbaren, zuverlaessigen Punkte). Faellt auf die Nase zurueck.
        """
        candidates = []
        for idx in (self.KP_NOSE, self.KP_LEFT_EAR, self.KP_RIGHT_EAR):
            if idx < len(person_kpts) and self._valid(person_kpts[idx]):
                candidates.append(person_kpts[idx][:2])
        if not candidates:
            return None
        pts = np.array(candidates, dtype=np.float32)
        return pts.mean(axis=0)

    # ------------------------------------------------------------------ #
    #  Hauptmethode: ein Update pro Frame                                 #
    # ------------------------------------------------------------------ #
    def update(self, keypoints):
        """
        Fuehrt fuer jeden verfolgten Punkt predict + (optional) update aus.

        :param keypoints: YOLO-Keypoints [Personen, Punkte, [x, y, conf]]
               (Rueckgabe von PoseEstimator.extract_keypoints) oder None.
        :return: dict mit pro Punkt: position (np.array[2]),
                 velocity (np.array[2]), speed (float), visible (bool).
        """
        # Messpunkte fuer diesen Frame bestimmen (erste erkannte Person)
        measurements = {"left_wrist": None, "right_wrist": None, "face": None}

        if keypoints is not None and len(keypoints) > 0:
            person = keypoints[0]
            # tensor -> numpy, falls noetig (z.B. torch.Tensor von Ultralytics)
            if hasattr(person, "cpu"):
                person = person.cpu().numpy()

            if len(person) > self.KP_LEFT_WRIST and self._valid(person[self.KP_LEFT_WRIST]):
                measurements["left_wrist"] = person[self.KP_LEFT_WRIST][:2]
            if len(person) > self.KP_RIGHT_WRIST and self._valid(person[self.KP_RIGHT_WRIST]):
                measurements["right_wrist"] = person[self.KP_RIGHT_WRIST][:2]
            measurements["face"] = self._face_center(person)

        # Pro Filter: erst vorhersagen, dann ggf. mit Messung korrigieren
        output = {}
        for name, kf in self.filters.items():
            kf.predict()  # Vorhersage laeuft IMMER (ueberbrueckt Aussetzer)

            meas = measurements[name]
            if meas is not None:
                kf.update(meas)
                self.misses[name] = 0
                visible = True
            else:
                # Keine Messung -> nur Vorhersage zaehlt als "miss"
                self.misses[name] += 1
                visible = self.misses[name] <= self.max_misses

            output[name] = {
                "position": kf.position(),
                "velocity": kf.velocity(),
                "speed": kf.speed(),
                "visible": visible and kf.initialized,
            }
        return output

    # ------------------------------------------------------------------ #
    #  Visualisierung                                                     #
    # ------------------------------------------------------------------ #
    def draw(self, frame, state=None):
        """
        Zeichnet die geglaetteten Positionen, Geschwindigkeitsvektoren und
        den Speed-Wert in das Bild. Praktisch fuer das Roadmap-Zwischenziel
        "Das System gibt live Bewegungswerte wie die Handgeschwindigkeit aus".

        :param frame: Bild, in das gezeichnet wird (wird modifiziert).
        :param state: optional das dict aus update(); sonst aktueller Stand.
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

            # Punkt
            cv2.circle(frame, (px, py), 6, color, -1)
            # Geschwindigkeitsvektor (5x skaliert zur Sichtbarkeit)
            cv2.arrowedLine(frame, (px, py),
                            (int(px + vx * 5), int(py + vy * 5)),
                            color, 2, tipLength=0.3)
            # Speed-Beschriftung
            cv2.putText(frame, f"{name}: {s['speed']:.1f} px/f",
                        (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, cv2.LINE_AA)

        return frame
