from statistics import mode
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import torch
from torchvision.datasets import ImageFolder
import json
import os


class animal10n(Dataset):
    def __init__(self, root_dir, transform, mode):
        if mode == 'train':
            self.train_set = ImageFolder(os.path.join(root_dir,mode), transform=transform)
        else:
            self.test_set = ImageFolder(os.path.join(root_dir,mode), transform=transform)
        self.mode = mode

    def __getitem__(self, index):
        if self.mode == 'train':
            img, target = self.train_set.__getitem__(index)
            return img, target, index
        else:
            img, target = self.test_set.__getitem__(index)
            return img, target, index

    def __delete__(self, index_list):
        if self.mode == 'train':
            for ind in sorted(index_list,reverse=True):
                self.train_set.samples.remove(self.train_set.samples[ind])
        else:
            for ind in sorted(index_list,reverse=True):
                self.test_set.samples.remove(self.test_set.samples[ind])
        

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_set.samples)
        else:
            return len(self.test_set.samples)

class animal10n_dataloader():
    def __init__(self, batch_size, num_workers, root_dir):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.T = transforms.Compose([
        transforms.Resize((32,32)),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def run(self, mode):
        if mode == 'train':
            train_dataset = animal10n(root_dir=self.root_dir, transform=self.T, mode="train")
            trainloader = DataLoader(
                dataset=train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)
            return train_dataset, trainloader

        elif mode == 'test':
            test_dataset = animal10n(root_dir=self.root_dir, transform=self.T, mode="test")
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return test_dataset, test_loader

# def main():
#     T = transforms.Compose([
#         transforms.Resize((32,32)),
#         #transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])

#     root = os.path.join(os.getcwd(),'dataset','animal10N','train')
#     loader = animal10n_dataloader(batch_size=128, num_workers=2, root_dir=root)
#     train_set, train_loader = loader.run('train')
#     train_set.__delete__([1,2,3,4,6,8])
#     train_loader = DataLoader(
#         dataset=train_set,
#         batch_size=train_loader.batch_size,
#         shuffle=True,
#         num_workers=train_loader.num_workers)

#     count = 0
#     for ind, (img, target,index) in enumerate(train_loader):
#         count += len(img)
#         print(count)
#     print(count)



# if __name__ == '__main__':
#     # freeze_support() here if program needs to be frozen
#     main()  # execute this only when run directly, not when imported!

