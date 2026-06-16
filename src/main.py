import cv2
from src.model_yolo import PoseEstimator
from src.face_processor import FaceProcessor
from src.model_resnet import ExpressionAnalyzer

# 7 Emotionen aus FER-2013
EMOTIONEN = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

def main():
    """
    Hauptprogramm: Orchestriert die Kamera, die Pose-Schätzung und die Gesichts-Isolierung.
    """
    # 1. Initialisierung der Komponenten
    pose_estimator = PoseEstimator()
    face_processor = FaceProcessor()
    emotion_analyzer = ExpressionAnalyzer(model_path="models/resnet_fer2013.pth")
    cap = cv2.VideoCapture(0) # Öffnet die Standard-Webcam

    print("Programm gestartet. Drücke 'q' zum Beenden.")

    while True:
        # Bild von der Kamera lesen
        ret, frame = cap.read()
        if not ret:
            print("Fehler: Kamera konnte nicht gelesen werden.")
            break
        
        
        # 1. Pose-Scheatzung durchführen
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
            class_id, confidence, _ = emotion_analyzer.predict(face_crop)
            emotion_text = f"{EMOTIONEN[class_id]} ({confidence:.0%})"

        # 5. Emotionen ins Hauptbild zeichnen
        cv2.putText(annotated_frame, emotion_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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
    print("Programm beendet.")

if __name__ == "__main__":
    main()
