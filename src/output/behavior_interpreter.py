import numpy as np
import cv2

EMOTIONEN = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


class BehaviorInterpreter:
    LEVEL_THRESHOLDS = [0.3, 0.7]
    LEVEL_NAMES = ["niedrig", "mittel", "hoch"]

    MATRIX = {
        ("angry", 0): "Groll / Kaltes Fressen",
        ("angry", 1): "Bestimmtheit / Protest",
        ("angry", 2): "Aggression / Wutausbruch",
        ("disgust", 0): "Stille Verachtung",
        ("disgust", 1): "Kritik / Ablehnung",
        ("disgust", 2): "Physischer Ekel (Repulsion)",
        ("fear", 0): "Schockstarre / Paralyse",
        ("fear", 1): "Sorge / Skepsis",
        ("fear", 2): "Panik / Hysterie",
        ("happy", 0): "Zufriedenheit / Gelassenheit",
        ("happy", 1): "Freundlichkeit / Charisma",
        ("happy", 2): "Euphorie / Ekstase",
        ("neutral", 0): "Fokus / Konzentration",
        ("neutral", 1): "Alltagsmodus / Baseline",
        ("neutral", 2): "Innere Unruhe / Maskierung",
        ("sad", 0): "Resignation / Depression",
        ("sad", 1): "Wehmut / Melancholie",
        ("sad", 2): "Verzweiflung / Agitierte Trauer",
        ("surprise", 0): "Verblüffung / Staunen",
        ("surprise", 1): "Interesse / Neugier",
        ("surprise", 2): "Erschrecken / Schrecksekunde",
    }

    def __init__(self, canvas_width=500, canvas_height=400):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height

    @staticmethod
    def _speed_idx(speed):
        if speed < 0.3:
            return 0
        if speed < 0.7:
            return 1
        return 2

    def interpret(self, emotion_probs, rel_speed):
        if max(emotion_probs) == 0:
            return {
                "has_face": False,
                "emotion": "",
                "confidence": 0.0,
                "speed_val": rel_speed,
                "speed_level": "",
                "speed_idx": -1,
                "label": "Kein Gesicht erkannt",
                "alternatives": [],
            }

        emotion_idx = int(np.argmax(emotion_probs))
        emotion = EMOTIONEN[emotion_idx]
        confidence = emotion_probs[emotion_idx]
        speed_idx = self._speed_idx(max(rel_speed, 0.0))

        return {
            "has_face": True,
            "emotion": emotion,
            "confidence": confidence,
            "speed_val": rel_speed,
            "speed_level": self.LEVEL_NAMES[speed_idx],
            "speed_idx": speed_idx,
            "label": self.MATRIX[(emotion, speed_idx)],
            "alternatives": [
                (self.LEVEL_NAMES[i], self.MATRIX[(emotion, i)])
                for i in range(3)
                if i != speed_idx
            ],
        }

    def render_canvas(self, result):
        W, H = self.canvas_width, self.canvas_height
        canvas = np.full((H, W, 3), 30, dtype=np.uint8)

        cv2.putText(canvas, "Behavior Interpretation",
                    (W // 2 - 160, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
        cv2.line(canvas, (20, 50), (W - 20, 50), (60, 60, 60), 1)

        if not result["has_face"]:
            cv2.putText(canvas, "Kein Gesicht erkannt",
                        (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
            return canvas

        label = result["label"]
        font_scale = 1.0 if len(label) < 20 else 0.8
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        cv2.putText(canvas, label,
                    (W // 2 - tw // 2, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 2)

        emotion_text = f"Emotion: {result['emotion']} ({result['confidence']:.0%})"
        cv2.putText(canvas, emotion_text,
                    (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        bar_x, bar_y, bar_w, bar_h = 20, 190, W - 40, 22
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (80, 80, 80), -1)

        fill_w = int(bar_w * min(result["speed_val"], 1.0))
        if result["speed_idx"] == 0:
            bar_color = (0, 180, 0)
        elif result["speed_idx"] == 1:
            bar_color = (0, 200, 200)
        else:
            bar_color = (0, 0, 200)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                      bar_color, -1)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (120, 120, 120), 1)

        speed_text = f"Bewegung: {result['speed_val']:.2f}  ({result['speed_level']})"
        cv2.putText(canvas, speed_text,
                    (20, bar_y + bar_h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        alt_y = bar_y + bar_h + 60
        cv2.putText(canvas, "Alternativ:",
                    (20, alt_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)
        for i, (level, alt_label) in enumerate(result["alternatives"]):
            text = f"  bei {level}er Bewegung: {alt_label}"
            cv2.putText(canvas, text,
                        (20, alt_y + 30 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        return canvas
