class GestureAnalyzer:
    """
    Analysiert einfache Körpergesten anhand der YOLO-Pose-Keypoints.
    """

    def analyze(self, keypoints):
        """
        Erkennt einfache Gesten der ersten erkannten Person.
        """

        if keypoints is None or len(keypoints) == 0:
            return "No Person"

        # Erste erkannte Person verwenden
        person = keypoints[0]

        # COCO-Keypoints:
        # 0 = Nase
        # 9 = linkes Handgelenk
        # 10 = rechtes Handgelenk
        nose = person[0]
        left_wrist = person[9]
        right_wrist = person[10]

        # y-Koordinate auslesen
        nose_y = nose[1]
        left_wrist_y = left_wrist[1]
        right_wrist_y = right_wrist[1]

        # Wenn beide Handgelenke über der Nase sind
        if left_wrist_y < nose_y and right_wrist_y < nose_y:
            return "Hands Up"

        # Wenn eine Hand über der Nase ist
        if left_wrist_y < nose_y or right_wrist_y < nose_y:
            return "One Hand Up"

        return "Neutral"