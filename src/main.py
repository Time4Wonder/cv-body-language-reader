import cv2
import json
from tracking_features.model_yolo import PoseEstimator
from tracking_features.face_processor import FaceProcessor
from spatial_analysis.model_resnet import ExpressionAnalyzer
from temporal_analysis.temporal_tracker import TemporalTracker
from temporal_analysis.temporal_aggregator import TemporalAggregator
from output.live_chart import LiveChart

# 7 Emotionen aus FER-2013
EMOTIONEN = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


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
    live_chart = LiveChart()
    cap = cv2.VideoCapture(0)  # Öffnet die Standard-Webcam

    image_size_set = False  # Bildgröße erst nach dem ersten Frame bekannt

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

        # 7. Zeitliche Aggregation (Scores über die Periode)
        # Bewegung = die schnellere der beiden Hände (px/frame)
        left_speed = motion_state["left_wrist"]["speed"]
        right_speed = motion_state["right_wrist"]["speed"]
        motion_speed = max(left_speed, right_speed)

        temporal_aggregator.add_frame(emotion_probs, motion_speed)
        live_chart.add_frame(emotion_probs, temporal_aggregator.normalize_speed(motion_speed))

        # --- ANZEIGE ---

        # Das kombinierte Bild (Skelett + Gesichtsrahmen) anzeigen
        cv2.imshow("Detection Overview", annotated_frame)

        # Den isolierten Gesichts-Ausschnitt in einem separaten Fenster zeigen
        if face_crop is not None:
            cv2.imshow("Face Crop (Model Input)", face_crop)

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
        print("Scores pro Emotion [0,1]:")
        print(json.dumps(session["emotion_scores"], indent=2))
        print(f"Bewegung (roh): {session['motion']['speed_raw']} px/frame")
    else:
        print("\nKeine Daten aufgezeichnet.")

    live_chart.close()
    print("Programm beendet.")


if __name__ == "__main__":
    main()
