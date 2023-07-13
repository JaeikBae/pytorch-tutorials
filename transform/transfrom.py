import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="dataset",
    train=True,
    download=True,
    transform=ToTensor()
)

target_trans = Lambda(lambda y: torch.zeros(
    10, dtype=torch.float).scatter_(
        dim=0, index=torch.tensor(y), value=1))