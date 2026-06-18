from cv2.gapi import threshold
import numpy as np


class BodyMovementAnalyzer:
    """
    Analysiert die Bewegungsintensität von Körper und Händen anhand der YOLO-Keypoints.
    Die Bewegung wird relativ zum Körperzentrum berechnet, damit Kamerabewegungen
    weniger Einfluss auf die Messung haben.
    """

    def __init__(self, max_body_movement=0.25, max_hand_movement=0.45, max_head_movement=0.30, smoothing=0.8):
        # Speichert die normalisierten Keypoints des vorherigen Frames
        self.previous_normalized_keypoints = None

        # Maximal erwartete Körperbewegung zur Normierung auf 0 bis 1
        self.max_body_movement = max_body_movement

        # Maximal erwartete Handbewegung zur Normierung auf 0 bis 1
        self.max_hand_movement = max_hand_movement

        # Maximal erwartete Kopfbewegung zur Normierung auf 0 bis 1
        self.max_head_movement = max_head_movement

        # Glättungsfaktor zur Reduzierung starker Schwankungen
        self.smoothing = smoothing

        # Geglättete Körperbewegung
        self.body_intensity = 0.0

        # Geglättete Handbewegung
        self.hand_intensity = 0.0

        # Geglättete Kopfbewegung
        self.head_intensity = 0.0

        

    def _normalize_keypoints(self, person):
        """
        Normalisiert die Keypoints relativ zum Körperzentrum und zur Körpergröße.
        Dadurch wird der Einfluss von Kamerabewegungen reduziert.
        """

        # COCO-Keypoints:
        # 5 = linke Schulter, 6 = rechte Schulter
        # 11 = linke Hüfte, 12 = rechte Hüfte
        left_shoulder = person[5][:2]
        right_shoulder = person[6][:2]
        left_hip = person[11][:2]
        right_hip = person[12][:2]

        # Körperzentrum aus Schultern und Hüfte berechnen
        body_center = np.mean(
            [left_shoulder, right_shoulder, left_hip, right_hip],
            axis=0
        )

        # Körpergröße als Skalierungsfaktor verwenden
        shoulder_center = np.mean([left_shoulder, right_shoulder], axis=0)
        hip_center = np.mean([left_hip, right_hip], axis=0)

        body_scale = np.linalg.norm(shoulder_center - hip_center)

        # Sicherheitscheck gegen Division durch 0
        if body_scale < 1:
            body_scale = 1

        # Keypoints relativ zum Körperzentrum normalisieren
        normalized_keypoints = (person[:, :2] - body_center) / body_scale

        return normalized_keypoints


    def _apply_deadzone(self, value, threshold=0.03):
        """
        Entfernt sehr kleine Bewegungen, die meistens durch YOLO-Rauschen entstehen.
        """
        if value < threshold:
            return 0.0
        return value
    
    def calculate(self, keypoints):
        """
        Berechnet getrennt die Körperbewegung und die Handbewegung.

        :param keypoints: YOLO-Keypoints der erkannten Personen
        :return: body_intensity, hand_intensity
        """

        if keypoints is None or len(keypoints) == 0:
            return 0.0, 0.0, 0.0

        # Erste erkannte Person verwenden
        person = keypoints[0].cpu().numpy()

        # Keypoints normalisieren, um Kamerabewegungen zu reduzieren
        current_normalized = self._normalize_keypoints(person)

        # Beim ersten Frame gibt es noch keinen Vergleich
        if self.previous_normalized_keypoints is None:
            self.previous_normalized_keypoints = current_normalized
            return 0.0, 0.0, 0.0

        previous_normalized = self.previous_normalized_keypoints

        # Abstand jedes Keypoints zum vorherigen Frame berechnen
        distances = np.linalg.norm(
            current_normalized - previous_normalized,
            axis=1
        )

        # Kopf-Keypoints separat behandeln
        head_indices = [0, 1, 2, 3, 4]

        # Körper-Keypoints ohne Kopf und ohne Hände
        body_indices = [5, 6, 11, 12, 13, 14, 15, 16]

        # Handgelenke separat behandeln
        hand_indices = [9, 10]
        
        # Durchschnittliche Kopfbewegung berechnen
        head_movement = self._apply_deadzone(np.mean(distances[head_indices]), 0.07)

        # Durchschnittliche Körperbewegung berechnen
        body_movement = self._apply_deadzone(np.mean(distances[body_indices]), 0.03)

        # Durchschnittliche Handbewegung berechnen
        hand_movement = self._apply_deadzone(np.mean(distances[hand_indices]), 0.02)

        # Werte auf 0 bis 1 normieren
        head_value = np.clip(head_movement / self.max_head_movement, 0.0, 1.0)
        body_value = np.clip(body_movement / self.max_body_movement, 0.0, 1.0)
        hand_value = np.clip(hand_movement / self.max_hand_movement, 0.0, 1.0)

        # Werte glätten
        self.head_intensity = (
            self.smoothing * self.head_intensity
            + (1 - self.smoothing) * head_value)

        self.body_intensity = (
            self.smoothing * self.body_intensity
            + (1 - self.smoothing) * body_value)

        self.hand_intensity = (
            self.smoothing * self.hand_intensity
            + (1 - self.smoothing) * hand_value)


        # Aktuelle Keypoints für den nächsten Frame speichern
        self.previous_normalized_keypoints = current_normalized

        return (
            float(self.body_intensity),
            float(self.hand_intensity),
            float(self.head_intensity)
        )
    