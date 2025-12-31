from torchvision import transforms
import numpy as np

class Augmentation:
    def __init__(self):
        self.augmentations = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=15),
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
        ])

    def apply(self, image):
        return self.augmentations(image)

    def apply_to_batch(self, images):
        return [self.apply(image) for image in images]