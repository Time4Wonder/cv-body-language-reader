import cv2
from model_yolo import PoseEstimator
from face_processor import FaceProcessor
from model_yolo import PoseEstimator, MovementAnalyzer, GestureAnalyzer

def main():
    """
    Hauptprogramm: Orchestriert die Kamera, die Pose-Schätzung und die Gesichts-Isolierung.
    """
    # 1. Initialisierung der Komponenten
    pose_estimator = PoseEstimator()
    face_processor = FaceProcessor()
    movement_analyzer = MovementAnalyzer()
    gesture_analyzer = GestureAnalyzer()
    
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
        
        # Bewegungsintensität berechnen
        movement_intensity = movement_analyzer.calculate_movement_intensity(keypoints)

        # Geste erkennen
        gesture = gesture_analyzer.analyze(keypoints)

        # Geste in numerische Features umwandeln
        gesture_features = gesture_analyzer.encode(gesture)

        # 2. Das Basis-Bild mit dem YOLO-Skelett erstellen
        # Wir nutzen 'annotated_frame' als unsere Zeichenfläche
        annotated_frame = pose_estimator.draw(results)
        if annotated_frame is None:
            annotated_frame = frame.copy()


        # 3. Das Gesicht isolieren und das Rechteck in unser Basis-Bild zeichnen
        face_crop = face_processor.extract_face(frame, keypoints)
        annotated_frame = face_processor.draw(annotated_frame)
        
        # --- ANZEIGE ---
        
        # Das kombinierte Bild (Skelett + Gesichtsrahmen) anzeigen
        cv2.imshow("Detection Overview", annotated_frame)
        # Bewegungsintensität anzeigen
        cv2.putText(
    annotated_frame,
    f"Movement Intensity: {movement_intensity:.2f}",
    (30, 50),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
)

        # Erkannte Geste anzeigen
        cv2.putText(annotated_frame,f"Gesture: {gesture}",(20, 60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 255, 0),1)
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