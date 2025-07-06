import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torchvision.transforms import (
        CenterCrop,
        Compose,
        RandomChoice,
        RandomCrop,
        RandomHorizontalFlip,
        RandomResizedCrop,
        Resize,
        ToTensor,
)

# 2. Create custom Dataset class
class CLICDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train'):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.image_files = [f for f in os.listdir(self.root_dir)
                           if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image


# class CLICDataset(Dataset):
#     def __init__(self, root_dir, latent_dir, transform=None, split='train'):
#         self.root_dir = os.path.join(root_dir, split)
#         self.latent_dir = os.path.join(latent_dir, split)
#         self.transform = transform

#         self.image_files = sorted([
#             f for f in os.listdir(self.root_dir)
#             if f.endswith(('.png', '.jpg', '.jpeg'))
#         ])

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         img_filename = self.image_files[idx]
#         img_path = os.path.join(self.root_dir, img_filename)
#         latent_path = os.path.join(self.latent_dir, os.path.splitext(img_filename)[0] + ".pt")

#         image = Image.open(img_path).convert('RGB')
#         if self.transform:
#             image = self.transform(image)

#         teacher_latent = torch.load(latent_path)

#         return image, teacher_latent


def build_trainloader(batch_size=32, img_dir="/home/iot/Desktop/IoT/datasets/CLIC/archive/val2017", latent_dir="/home/iot/Desktop/IoT/ILLM_VLO1_Train_Latents"):
    def default_train_transform(image_size: int) -> Compose:
        choice_transform = RandomChoice(
            [
                RandomCrop(size=image_size, pad_if_needed=True, padding_mode="reflect"),
                RandomResizedCrop(size=image_size),
            ]
        )
        return Compose(
            [
                choice_transform,
                RandomHorizontalFlip(),
                ToTensor(),
            ]
        )

    transform = default_train_transform(224)
    
    train_dataset = CLICDataset(root_dir=os.path.join(img_dir), transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=0)

    return train_loader

