import torch
from PIL import Image
import torchvision.transforms as transforms

class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset .
    """
    path = '/content/train_images/'
    transform = transforms.Compose([
        transforms.PILToTensor()
])

    def __init__(self, items, labels):
        self.items = items
        self.labels = labels

    def __getitem__(self, index):
        item = self.items[index]
        label = self.labels[index]

        temp_path = StartingDataset.path + self.items[index]
        image = Image.open(temp_path)
        image_tensor = StartingDataset.transform(image)
        return image_tensor, label

    def __len__(self):
        return len(self.labels)
