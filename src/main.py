import cv2
from model_yolo import PoseEstimator
from face_processor import FaceProcessor
from BodyMovementAnalyzer import BodyMovementAnalyzer

def main():
    """
    Hauptprogramm: Orchestriert die Kamera, die Pose-Schätzung und die Gesichts-Isolierung.
    """
    # 1. Initialisierung der Komponenten
    pose_estimator = PoseEstimator()
    face_processor = FaceProcessor()
    body_movement_analyzer = BodyMovementAnalyzer()
    
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
        body_movement, hand_movement, head_movement = body_movement_analyzer.calculate(keypoints)
    
        # 2. Das Basis-Bild mit dem YOLO-Skelett erstellen
        # Wir nutzen 'annotated_frame' als unsere Zeichenfläche
        annotated_frame = pose_estimator.draw(results)
        if annotated_frame is None:
            annotated_frame = frame.copy()


        # 3. Das Gesicht isolieren und das Rechteck in unser Basis-Bild zeichnen
        face_crop = face_processor.extract_face(frame, keypoints)
        annotated_frame = face_processor.draw(annotated_frame)
        
        # --- ANZEIGE ---
        
        # Bewegungsintensität anzeigen
        cv2.putText(annotated_frame, f"Body Movement: {body_movement:.2f}",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(annotated_frame, f"Hand Movement: {hand_movement:.2f}",
            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.putText(annotated_frame, f"Head Movement: {head_movement:.2f}",
            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        
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