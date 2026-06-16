import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


class ExpressionAnalyzer:
    # Lädt ein vortrainiertes ResNet18 und ersetzt den Kopf durch 7 Emotionen
    def __init__(self, num_classes=7, model_path=None):
        self.model = models.resnet18(weights="DEFAULT")
        in_features = self.model.fc.in_features # Head entfernen
        self.model.fc = nn.Linear(in_features, num_classes)
        self.model.eval()
        
        if model_path: 
            self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
            self.model.eval()


    # Wandelt ein OpenCV-BGR-Bild in einen normalisierten Tensor um (1, 3, 224, 224)
    def preprocess(self, face_crop):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(face_crop)

    # Sagt 7 Emotionen vorher, gibt (class_id, confidence, alle_probs) zurück
    def predict(self, face_crop):
        # OpenCV liefert BGR, ResNet erwartet RGB
        rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        tensor = self.preprocess(rgb).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
        class_id = probs.argmax().item()
        confidence = probs[0, class_id].item()
        return class_id, confidence, probs[0].tolist()
