import time
import numpy as np

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


class _WindowAccumulator:
    """
    Sammelt Frame-Werte und berechnet aggregierte Kennzahlen.
    """

    def __init__(self):
        self.emotion_sum = np.zeros(len(EMOTIONS), dtype=np.float64)
        self.emotion_peak = np.zeros(len(EMOTIONS), dtype=np.float64)
        self.dominant_count = np.zeros(len(EMOTIONS), dtype=np.float64)
        self.motion_sum = 0.0
        self.motion_peak = 0.0
        self.n_frames = 0
        self.t_start = None
        self.t_end = None

    def add(self, emotion_probs, motion_speed, timestamp):
        """
        Füge Werte eines Frames hinzu.
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
        Berechne aggregierte Kennzahlen.
        """
        if self.n_frames == 0:
            return None

        n = self.n_frames
        emotion_score = self.emotion_sum / n
        dominant_fraction = self.dominant_count / n
        motion_raw = self.motion_sum / n
        motion_norm = float(np.clip(motion_raw / (image_diagonal + 1e-6), 0.0, 1.0))

        duration = (self.t_end - self.t_start) if self.t_start is not None else 0.0

        return {
            "n_frames": n,
            "duration_s": round(duration, 2),
            "emotions": {
                EMOTIONS[i]: {
                    "score": round(float(emotion_score[i]), 4),
                    "peak": round(float(self.emotion_peak[i]), 4),
                    "dominant_fraction": round(float(dominant_fraction[i]), 4),
                }
                for i in range(len(EMOTIONS))
            },
            "emotion_scores": {
                EMOTIONS[i]: round(float(emotion_score[i]), 4)
                for i in range(len(EMOTIONS))
            },
            "dominant_emotion": EMOTIONS[int(np.argmax(emotion_score))],
            "motion": {
                "speed_raw": round(motion_raw, 3),
                "speed_norm": round(motion_norm, 4),
                "speed_peak_raw": round(self.motion_peak, 3),
            },
        }


class TemporalAggregator:
    """
    Aggregiere Frame-Signale ueber die Zeit.
    """

    def __init__(self, window_seconds=10.0, image_size=None):
        """
        Initialisiere den Aggregator.
        """
        self.window_seconds = float(window_seconds)
        self.image_diagonal = None
        if image_size is not None:
            self.set_image_size(*image_size)
        self._session = _WindowAccumulator()
        self._current_slice = _WindowAccumulator()
        self._slice_start_t = None
        self._completed_slices = []

    def set_image_size(self, width, height):
        """Setzt die Bildgroesse (zur Bewegungs-Normalisierung)."""
        self.image_diagonal = float(np.hypot(width, height))

    def add_frame(self, emotion_probs, motion_speed, timestamp=None):
        """
        Verarbeite einen Frame.
        """
        if timestamp is None:
            timestamp = time.time()
        diag = self.image_diagonal if self.image_diagonal else 1.0
        self._session.add(emotion_probs, motion_speed, timestamp)
        self._current_slice.add(emotion_probs, motion_speed, timestamp)
        if self._slice_start_t is None:
            self._slice_start_t = timestamp
        if (timestamp - self._slice_start_t) >= self.window_seconds:
            res = self._current_slice.result(diag)
            if res is not None:
                res["slice_index"] = len(self._completed_slices)
                self._completed_slices.append(res)
            self._current_slice = _WindowAccumulator()
            self._slice_start_t = timestamp

    def get_slices(self, include_current=True):
        """
        Gib die Zeit-Slice-Ergebnisse.
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
        Gib die Session-Zusammenfassung.
        """
        diag = self.image_diagonal if self.image_diagonal else 1.0
        return self._session.result(diag)

    def get_report(self):
        """
        Gib den kompletten Report.
        """
        return {
            "window_seconds": self.window_seconds,
            "session": self.get_session_summary(),
            "slices": self.get_slices(include_current=True),
        }
