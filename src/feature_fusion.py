import numpy as np


class FeatureFusion:
    """
    Kombiniert verschiedene Merkmale zu einem gemeinsamen Feature-Vektor.
    """

    def fuse(self, movement_intensity, gesture_features=None, resnet_features=None):
        """
        Erstellt einen gemeinsamen Feature-Vektor für den späteren ML-Classifier.
        """

        features = []

        # Bewegungsintensität hinzufügen
        features.append(movement_intensity)

        # Gestik-Merkmale hinzufügen, falls vorhanden
        if gesture_features is not None:
            features.extend(gesture_features)

        # ResNet-Merkmale hinzufügen, falls vorhanden
        if resnet_features is not None:
            features.extend(resnet_features)

        # In NumPy-Array umwandeln
        return np.array(features, dtype=np.float32)