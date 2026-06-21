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

    def __init__(self, canvas_width=500, canvas_height=580):
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

    def _draw_bar(self, canvas, x, y, w, h, fill, speed_idx):
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (80, 80, 80), -1)
        fill_w = int(w * min(fill, 1.0))
        if speed_idx == 0:
            bar_color = (0, 180, 0)
        elif speed_idx == 1:
            bar_color = (0, 200, 200)
        else:
            bar_color = (0, 0, 200)
        cv2.rectangle(canvas, (x, y), (x + fill_w, y + h), bar_color, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (120, 120, 120), 1)

    def _draw_main_section(self, canvas, result, W):
        if not result["has_face"]:
            cv2.putText(canvas, "Kein Gesicht erkannt",
                        (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (100, 100, 100), 2)
            return

        label = result["label"]
        font_scale = 1.0 if len(label) < 20 else 0.8
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                      font_scale, 2)
        cv2.putText(canvas, label, (W // 2 - tw // 2, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 2)

        cv2.putText(canvas,
                    f"Emotion: {result['emotion']} ({result['confidence']:.0%})",
                    (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1)

        self._draw_bar(canvas, 20, 190, W - 40, 22,
                       result["speed_val"], result["speed_idx"])

        cv2.putText(canvas,
                    f"Bewegung: {result['speed_val']:.2f}  ({result['speed_level']})",
                    (20, 237), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1)

        cv2.putText(canvas, "Alternativ:", (20, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)
        for i, (level, alt_label) in enumerate(result["alternatives"]):
            cv2.putText(canvas, f"  bei {level}er Bewegung: {alt_label}",
                        (20, 310 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (180, 180, 180), 1)

    def _draw_avg_section(self, canvas, avg_result, W):
        sep_y = 360
        cv2.line(canvas, (20, sep_y), (W - 20, sep_y), (60, 60, 60), 1)

        cv2.putText(canvas, "----- 5s ----", (W // 2 - 65, sep_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        y = sep_y + 55
        avg_label = avg_result["label"]
        font_scale = 1.0 if len(avg_label) < 20 else 0.8
        (tw, _), _ = cv2.getTextSize(avg_label, cv2.FONT_HERSHEY_SIMPLEX,
                                      font_scale, 2)
        cv2.putText(canvas, avg_label, (W // 2 - tw // 2, y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 2)

        y += 30
        cv2.putText(canvas,
                    f" Emotion: {avg_result['emotion']} ({avg_result['confidence']:.0%})",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (200, 200, 200), 1)

        y += 25
        self._draw_bar(canvas, 20, y, W - 40, 22,
                       avg_result["speed_val"], avg_result["speed_idx"])

        cv2.putText(canvas,
                    f" Bewegung: {avg_result['speed_val']:.2f} ({avg_result['speed_level']})",
                    (20, y + 22 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (180, 180, 180), 1)

    def render_canvas(self, result, avg_result=None):
        W, H = self.canvas_width, self.canvas_height
        canvas = np.full((H, W, 3), 30, dtype=np.uint8)

        cv2.putText(canvas, "Behavior Interpretation",
                    (W // 2 - 160, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
        cv2.line(canvas, (20, 50), (W - 20, 50), (60, 60, 60), 1)

        self._draw_main_section(canvas, result, W)

        if avg_result is not None and result["has_face"]:
            self._draw_avg_section(canvas, avg_result, W)

        return canvas
