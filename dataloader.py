# coding: utf-8

import numpy as np

import torch
import torch.utils.data

import torchvision
import torchvision.models
import torchvision.transforms


def get_loader(batch_size, num_workers):
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])

    transform = transforms.Compose([transforms.Resize((80,120)),
                                    transforms.CenterCrop((64,85)),
                                    transforms.Grayscale(),
                                    transforms.ToTensor(),
                                    transforms.Normalize( mean, std )])
    # train_transform = torchvision.transforms.Compose([
    #     torchvision.transforms.RandomCrop(32, padding=4),
    #     torchvision.transforms.RandomHorizontalFlip(),
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize(mean, std),
    # ])
    # test_transform = torchvision.transforms.Compose([
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize(mean, std),
    # ])
    training_dataset = datasets.ImageFolder(root='content/dataset/training_set',transform=transform)
    validation_dataset = datasets.ImageFolder(root='content/dataset/test_set',transform=transform)

    training_loader =  torch.utils.data.DataLoader(dataset=training_dataset,  batch_size=100, num_workers=num_workers, shuffle=True,drop_last=True)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset,  batch_size=100, num_workers=num_workers, shuffle=True, drop_last=True)


    return training_loader, validation_loader
