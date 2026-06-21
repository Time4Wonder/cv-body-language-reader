import time
import json
from collections import deque

import numpy as np
import cv2
from tracking_features.model_yolo import PoseEstimator
from tracking_features.face_processor import FaceProcessor
from spatial_analysis.model_resnet import ExpressionAnalyzer
from temporal_analysis.temporal_tracker import TemporalTracker
from temporal_analysis.temporal_aggregator import TemporalAggregator
from temporal_analysis.relative_motion import RelativeMotionAnalyzer
from output.live_chart import LiveChart
from output.behavior_interpreter import BehaviorInterpreter

# 7 Emotionen aus FER-2013
EMOTIONEN = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
BUFFER_SECONDS = 5.0  # Zeitfenster für rolling average


def main():
    """
    Hauptprogramm: Orchestriert die Kamera, die Pose-Schätzung, die
    Gesichts-Isolierung, das zeitliche Tracking und die zeitliche Aggregation.
    """
    # 1. Initialisierung der Komponenten
    pose_estimator = PoseEstimator()
    face_processor = FaceProcessor()
    emotion_analyzer = ExpressionAnalyzer(model_path="models/resnet_fer2013.pth")
    temporal_tracker = TemporalTracker()
    temporal_aggregator = TemporalAggregator(window_seconds=10.0)
    relative_motion = RelativeMotionAnalyzer()
    behavior_interpreter = BehaviorInterpreter()
    live_chart = LiveChart()
    cap = cv2.VideoCapture(0)  # Öffnet die Standard-Webcam

    image_size_set = False  # Bildgröße erst nach dem ersten Frame bekannt
    behavior_speeds = []    # rel_speed-Werte für Session-Report
    history = deque()       # Rolling Buffer (timestamp, emotion_probs, rel_speed)

    print("Programm gestartet. Drücke 'q' zum Beenden.")

    while True:
        # Bild von der Kamera lesen
        ret, frame = cap.read()
        if not ret:
            print("Fehler: Kamera konnte nicht gelesen werden.")
            break

        # Bildgröße einmalig setzen (für die Bewegungs-Normalisierung auf [0,1])
        if not image_size_set:
            h, w, _ = frame.shape
            temporal_aggregator.set_image_size(w, h)
            image_size_set = True

        # 1. Pose-Schätzung durchführen
        results = list(pose_estimator.estimate(frame))
        keypoints = pose_estimator.extract_keypoints(results)

        # 2. Das Basis-Bild mit dem YOLO-Skelett erstellen
        # Wir nutzen 'annotated_frame' als unsere Zeichenfläche
        annotated_frame = pose_estimator.draw(results)
        if annotated_frame is None:
            annotated_frame = frame.copy()

        # 3. Das Gesicht isolieren und das Rechteck in unser Basis-Bild zeichnen
        face_crop = face_processor.extract_face(frame, keypoints)
        annotated_frame = face_processor.draw(annotated_frame)

        # 4. Emotionen auf dem Gesichts-Crop vorhersagen
        emotion_text = "Kein Gesicht"
        if face_crop is not None:
            class_id, confidence, emotion_probs = emotion_analyzer.predict(face_crop)
            emotion_text = f"{EMOTIONEN[class_id]} ({confidence:.0%})"
        else:
            emotion_probs = [0.0] * 7

        # 5. Emotionen ins Hauptbild zeichnen
        cv2.putText(annotated_frame, emotion_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 6. Zeitliche Analyse (Kalman-Tracking)
        motion_state = temporal_tracker.update(keypoints)
        annotated_frame = temporal_tracker.draw(annotated_frame, motion_state)

        # 7. Relative Handbewegung (Kalman-gefiltert, relativ zum Gesicht, [0,1])
        rel_speed = relative_motion.compute(motion_state, keypoints)

        behavior_result = behavior_interpreter.interpret(emotion_probs, rel_speed)

        # 8. Rolling 5s-Durchschnitt
        now = time.time()
        history.append((now, emotion_probs, rel_speed))
        while history and now - history[0][0] > BUFFER_SECONDS:
            history.popleft()

        if len(history) > 0:
            avg_probs = np.mean([h[1] for h in history], axis=0).tolist()
            avg_speed = float(np.mean([h[2] for h in history]))
            avg_result = behavior_interpreter.interpret(avg_probs, avg_speed)
        else:
            avg_result = None

        behavior_canvas = behavior_interpreter.render_canvas(behavior_result, avg_result)

        behavior_speeds.append(rel_speed)
        temporal_aggregator.add_frame(emotion_probs, rel_speed)
        live_chart.add_frame(emotion_probs, rel_speed)

        # --- ANZEIGE ---

        # Das kombinierte Bild (Skelett + Gesichtsrahmen) anzeigen
        cv2.imshow("Detection Overview", annotated_frame)

        # Den isolierten Gesichts-Ausschnitt in einem separaten Fenster zeigen
        if face_crop is not None:
            cv2.imshow("Face Crop (Model Input)", face_crop)

        cv2.imshow("Behavior Interpretation", behavior_canvas)

        # Abbruchbedingung: 'q' Taste
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Ressourcen sauber freigeben
    cap.release()
    cv2.destroyAllWindows()

    # 8. Abschlussbericht: aggregierte Scores über die gesamte Session
    report = temporal_aggregator.get_report()
    session = report["session"]
    if session is not None:
        print("\n=== Zusammenfassung der Session ===")
        print(f"Dauer: {session['duration_s']}s, Frames: {session['n_frames']}")
        print(f"Dominante Emotion: {session['dominant_emotion']}")

        avg_speed = sum(behavior_speeds) / len(behavior_speeds) if behavior_speeds else 0
        avg_probs = [session["emotion_scores"][e] for e in EMOTIONEN]

        print("\n─── Emotions-Scores [0,1] ───")
        for i, emotion in enumerate(EMOTIONEN):
            score = session["emotion_scores"][emotion]
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            print(f"  {emotion:10s} |{bar}| {score:.2f}")

        print(f"\n  Bewegung (roh): {session['motion']['speed_raw']:.3f} px/frame")
        print(f"  Bewegung (norm): {session['motion']['speed_norm']:.2f}")

        session_behavior = behavior_interpreter.interpret(avg_probs, avg_speed)
        print(f"\n  Session-Interpretation: {session_behavior['label']}")
    else:
        print("\nKeine Daten aufgezeichnet.")

    live_chart.close()
    print("Programm beendet.")


if __name__ == "__main__":
    main()
