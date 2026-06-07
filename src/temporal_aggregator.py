import time
import numpy as np

# =============================================================================
#  Zeitliche Aggregation (Temporal Aggregation)
# -----------------------------------------------------------------------------
#  Aufgabe (Roadmap Phase 4 & 5):
#    Aus den Frame-fuer-Frame-Signalen (CNN-Emotionen + Bewegung aus dem
#    TemporalTracker) ueber eine Zeitperiode hinweg pro Emotion EINEN Score
#    in [0, 1] berechnen -> Grundlage fuer Graph und Report.
#
#  Trennung der Verantwortlichkeiten (3 Ebenen):
#    1. model_yolo.py / CNN  -> Wahrnehmung pro Frame
#    2. temporal_tracker.py  -> Glaettung pro Frame (Kalman)
#    3. temporal_aggregator  -> Aggregation ueber eine PERIODE (diese Datei)
#
#  "Scaling der Zeit" = Normalisierung durch die Dauer:
#    Wir mitteln die Wahrscheinlichkeiten ueber alle Frames eines Fensters.
#    Da CNN-Softmax-Werte bereits in [0, 1] liegen, ist der Mittelwert
#    automatisch in [0, 1] und UNABHAENGIG von der Sessiondauer.
# =============================================================================

# Reihenfolge der 7 Klassen wie im Trainings-Datensatz (FER / AffectNet-Style)
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


class _WindowAccumulator:
    """
    Sammelt rohe Frame-Werte fuer EIN Zeitfenster und berechnet daraus die
    aggregierten Kennzahlen. Wird sowohl fuer Zeit-Slices als auch fuer die
    Gesamtsession (ein einziges grosses Fenster) verwendet.
    """

    def __init__(self):
        # Summe der Emotions-Wahrscheinlichkeiten ueber alle Frames (7-Vektor)
        self.emotion_sum = np.zeros(len(EMOTIONS), dtype=np.float64)
        # Maximum pro Emotion (fuer "peak": gab es einen kurzen, starken Ausschlag?)
        self.emotion_peak = np.zeros(len(EMOTIONS), dtype=np.float64)
        # Zaehler, wie oft jede Emotion die dominante (argmax) war
        self.dominant_count = np.zeros(len(EMOTIONS), dtype=np.float64)

        # Bewegung: Summe der (rohen) Handgeschwindigkeit in px/frame
        self.motion_sum = 0.0
        self.motion_peak = 0.0

        self.n_frames = 0
        self.t_start = None
        self.t_end = None

    def add(self, emotion_probs, motion_speed, timestamp):
        """
        Fuegt die Werte eines Frames hinzu.
        :param emotion_probs: 7-Vektor mit Wahrscheinlichkeiten (Summe ~ 1).
        :param motion_speed: skalare Handgeschwindigkeit in px/frame (roh).
        :param timestamp: Zeitstempel des Frames (time.time()).
        """
        p = np.asarray(emotion_probs, dtype=np.float64)
        self.emotion_sum += p
        self.emotion_peak = np.maximum(self.emotion_peak, p)
        self.dominant_count[int(np.argmax(p))] += 1.0

        self.motion_sum += float(motion_speed)
        self.motion_peak = max(self.motion_peak, float(motion_speed))

        if self.t_start is None:
            self.t_start = timestamp
        self.t_end = timestamp
        self.n_frames += 1

    def result(self, image_diagonal):
        """
        Berechnet die aggregierten Kennzahlen fuer dieses Fenster.

        :param image_diagonal: Bilddiagonale in px, zur Normalisierung der
               Bewegung auf [0, 1] (robust gegenueber Aufloesung).
        :return: dict mit allen Scores.
        """
        if self.n_frames == 0:
            return None

        n = self.n_frames

        # --- Emotions-Score in [0, 1]: Mittelwert ueber die Frames ---------
        # Zeit-Normalisierung: Teilen durch die Frame-Anzahl -> dauerunabhaengig.
        emotion_score = self.emotion_sum / n  # 7-Vektor in [0, 1]

        # Anteil der Zeit, in der die Emotion dominant war (ebenfalls [0, 1])
        dominant_fraction = self.dominant_count / n

        # --- Bewegung -------------------------------------------------------
        motion_raw = self.motion_sum / n  # px/frame (roh, fuer CSV/Training)
        # Normalisierte Version NUR fuer Graph/Report:
        motion_norm = float(np.clip(motion_raw / (image_diagonal + 1e-6), 0.0, 1.0))

        duration = (self.t_end - self.t_start) if self.t_start is not None else 0.0

        return {
            "n_frames": n,
            "duration_s": round(duration, 2),
            # Pro Emotion: name -> {score, peak, dominant_fraction}
            "emotions": {
                EMOTIONS[i]: {
                    "score": round(float(emotion_score[i]), 4),
                    "peak": round(float(self.emotion_peak[i]), 4),
                    "dominant_fraction": round(float(dominant_fraction[i]), 4),
                }
                for i in range(len(EMOTIONS))
            },
            # Bequemer Direktzugriff: nur die [0,1]-Scores als dict
            "emotion_scores": {
                EMOTIONS[i]: round(float(emotion_score[i]), 4)
                for i in range(len(EMOTIONS))
            },
            "dominant_emotion": EMOTIONS[int(np.argmax(emotion_score))],
            "motion": {
                "speed_raw": round(motion_raw, 3),       # px/frame (roh)
                "speed_norm": round(motion_norm, 4),     # [0, 1] fuer Graph
                "speed_peak_raw": round(self.motion_peak, 3),
            },
        }


class TemporalAggregator:
    """
    Aggregiert die Frame-Signale ueber die Zeit und liefert:
      - Zeit-Slices (z.B. alle 10 s) -> Achse fuer den zeitlichen Graphen
      - Gesamtsession                -> ein Score-Satz pro Emotion fuer den Report

    Beide werden in EINEM Durchlauf berechnet. Der Gesamtwert ergibt sich aus
    denselben Rohdaten wie die Slices (kein doppeltes Rechnen noetig).
    """

    def __init__(self, window_seconds=10.0, image_size=None):
        """
        :param window_seconds: Laenge eines Zeit-Slices in Sekunden.
               Bei 30s-Sessions ggf. kleiner waehlen (z.B. 5), bei langen
               Sessions ist 10 ein guter Standard.
        :param image_size: (breite, hoehe) des Kamerabildes, fuer die
               Normalisierung der Bewegung. Kann spaeter via set_image_size()
               gesetzt werden, sobald der erste Frame bekannt ist.
        """
        self.window_seconds = float(window_seconds)
        self.image_diagonal = None
        if image_size is not None:
            self.set_image_size(*image_size)

        self._session = _WindowAccumulator()       # gesamte Session
        self._current_slice = _WindowAccumulator()  # aktuelles Zeitfenster
        self._slice_start_t = None
        self._completed_slices = []                  # Liste fertiger Slice-Results

    def set_image_size(self, width, height):
        """Setzt die Bildgroesse (zur Bewegungs-Normalisierung)."""
        self.image_diagonal = float(np.hypot(width, height))

    def add_frame(self, emotion_probs, motion_speed, timestamp=None):
        """
        Verarbeitet EINEN Frame. In der main-Loop pro Durchlauf aufrufen.

        :param emotion_probs: 7-Vektor (CNN-Softmax) in Reihenfolge EMOTIONS.
               Falls in dieser Session noch kein CNN existiert, kann ein
               Null-/Gleichverteilungsvektor uebergeben werden.
        :param motion_speed: skalare Handgeschwindigkeit (px/frame), z.B.
               max(left_wrist.speed, right_wrist.speed) aus dem TemporalTracker.
        :param timestamp: optionaler Zeitstempel; sonst time.time().
        """
        if timestamp is None:
            timestamp = time.time()

        # Fallback fuer die Bilddiagonale, falls noch nicht gesetzt
        diag = self.image_diagonal if self.image_diagonal else 1.0

        # In Session UND aktuellen Slice einspeisen
        self._session.add(emotion_probs, motion_speed, timestamp)
        self._current_slice.add(emotion_probs, motion_speed, timestamp)

        if self._slice_start_t is None:
            self._slice_start_t = timestamp

        # Slice abschliessen, sobald window_seconds ueberschritten sind
        if (timestamp - self._slice_start_t) >= self.window_seconds:
            res = self._current_slice.result(diag)
            if res is not None:
                res["slice_index"] = len(self._completed_slices)
                self._completed_slices.append(res)
            self._current_slice = _WindowAccumulator()
            self._slice_start_t = timestamp

    def get_slices(self, include_current=True):
        """
        Gibt die Liste der Zeit-Slice-Ergebnisse zurueck (Achse fuer Graph).
        :param include_current: auch das noch nicht volle aktuelle Fenster?
        """
        diag = self.image_diagonal if self.image_diagonal else 1.0
        slices = list(self._completed_slices)
        if include_current:
            cur = self._current_slice.result(diag)
            if cur is not None:
                cur["slice_index"] = len(slices)
                slices.append(cur)
        return slices

    def get_session_summary(self):
        """
        Gibt den aggregierten Score-Satz fuer die GESAMTE Session zurueck.
        Das ist die Ein-Zeilen-Zusammenfassung pro Emotion fuer den Report.
        """
        diag = self.image_diagonal if self.image_diagonal else 1.0
        return self._session.result(diag)

    def get_report(self):
        """
        Komplettes, serialisierbares Ergebnis fuer Graph + Report (Phase 5).
        :return: dict { "session": {...}, "slices": [ {...}, ... ] }
        """
        return {
            "window_seconds": self.window_seconds,
            "session": self.get_session_summary(),
            "slices": self.get_slices(include_current=True),
        }
