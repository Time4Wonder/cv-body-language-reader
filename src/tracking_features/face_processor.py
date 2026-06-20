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
            x_l_ear, conf_l = int(left_ear[0]), left_ear[2].item()
            x_r_ear, conf_r = int(right_ear[0]), right_ear[2].item()

            # 2. Kopfbreite aus sichtbaren Ohren schätzen
            if conf_l > 0.5 and conf_r > 0.5:
                head_width = abs(x_r_ear - x_l_ear)
            elif conf_l > 0.5:
                head_width = abs(x_nose - x_l_ear) * 2
            elif conf_r > 0.5:
                head_width = abs(x_nose - x_r_ear) * 2
            else:
                head_width = int(frame.shape[0] * 0.3)

            head_width = max(head_width, 20)
            size = int(head_width * 1.6)
            size = max(size, 40)

            # 3. Quadratischen Ausschnitt um die Nase
            x1 = x_nose - size // 2
            y1 = y_nose - size // 2
            x2 = x_nose + size // 2
            y2 = y_nose + size // 2

            # 4. Grenzprüfung: Verhindern, dass außerhalb des Bildrahmens geschnitten wird
            h, w, _ = frame.shape
            y1, y2 = max(0, y1), min(h, y2)
            x1, x2 = max(0, x1), min(w, x2)

            # Koordinaten für die spätere draw()-Funktion speichern
            self.x1, self.y1 = x1, y1
            self.x2, self.y2 = x2, y2

            # 5. Den Ausschnitt durchführen und auf 224×224 skalieren
            crop_image = frame[y1:y2, x1:x2]

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