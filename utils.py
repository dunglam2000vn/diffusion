import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np


def get_data(img_height, img_width, batch_size, data_path):
    transform = transforms.Compose([
        #transforms.Resize((IMG_HEIGHT,IMG_WIDTH)),
        transforms.CenterCrop((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x*2 - 1)
    ])

    train_dataset = torchvision.datasets.CelebA(data_path, transform=transform, download=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                            shuffle=True)
    
    return train_dataset, train_loader

def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Display the first image of the batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 

    plt.imshow(reverse_transforms(image))