from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader


def get_dataloaders(batch_size=32):
    # Vorgaben fürs Transfomieren unsere Daten:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    ) 
    train_dataset = ImageFolder(root="data/raw/train", transform=transform)
    test_dataset = ImageFolder(root="data/raw/test", transform=transform)

    train_loader= DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

