import cv2

class FaceProcessor:
    def __init__(self, size=224):
        """
        Initialisiert den FaceProcessor mit einer Zielgröße für das Ausgabebild.
        """
        self.keypoints = None
        self.size = size

    def extract_face(self, frame, keypoints):
        """
        Extrahiert die Gesichtsregion aus dem Bild basierend auf den YOLO-Pose-Keypoints.
        Gibt ein skaliertes, quadratisches Bild des Gesichts zurück oder None, falls es fehlschlägt.
        """
        self.keypoints = keypoints
        
        # Sicherheitscheck: Sicherstellen, dass Keypoints vorhanden sind
        if self.keypoints is None or len(self.keypoints) == 0:
            return None

        try:
            # 1. Wichtige Merkmale für die erste erkannte Person identifizieren
            # COCO Keypoints: 0=Nase, 3=linkes_Ohr, 4=rechtes_Ohr
            nose = self.keypoints[0][0]
            left_ear = self.keypoints[0][3]
            right_ear = self.keypoints[0][4]

            # Koordinaten in Ganzzahlen (Pixel) umwandeln
            x_nose, y_nose = int(nose[0]), int(nose[1])
            x_l_ear = int(left_ear[0])
            x_r_ear = int(right_ear[0])

            # 2. Die Grenzen für den Ausschnitt festlegen
            # Vertikal: Nase als Zentrum mit festem Versatz nutzen
            y1, y2 = y_nose - 150, y_nose + 150
            
            # Horizontal: Ohren nutzen, um die Breite mit etwas Puffer zu definieren
            x1 = min(x_l_ear, x_r_ear) - 40
            x2 = max(x_l_ear, x_r_ear) + 40

            # 3. Grenzprüfung: Verhindern, dass außerhalb des Bildrahmens geschnitten wird
            h, w, _ = frame.shape
            y1, y2 = max(0, y1), min(h, y2)
            x1, x2 = max(0, x1), min(w, x2)

            # Koordinaten für die spätere draw()-Funktion speichern
            self.x1, self.y1 = x1, y1
            self.x2, self.y2 = x2, y2
            
            # 4. Den Ausschnitt durchführen
            crop_image = frame[y1:y2, x1:x2]

            # 5. Skalieren und zurückgeben
            if crop_image.size > 0:
                return cv2.resize(crop_image, (self.size, self.size))
            
        except Exception as e:
            print(f"Fehler bei der Gesichtsextraktion: {e}")
            
        return None
    
    def draw(self, frame):
        """Zeichnet ein Rechteck um das erkannte Gesicht auf das übergebene Bild."""
        if hasattr(self, "x1"):
            # Ein blaues Rechteck mit Dicke 2 zeichnen
            cv2.rectangle(frame, (self.x1, self.y1), (self.x2, self.y2), (255, 0, 0), 2)
        
        return frame