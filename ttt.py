import numpy as np
import random
import time
import math
import csv
import cv2
import os
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torchsummary import summary
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from get_kmer import Kmer_extractor
# from kmernn import test, train

def load_data(path="MNIST_data/mnist.npz"):
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)

def mkdir(path):
    # 判斷目錄是否存在, 存在: True ; 不存在: False
    folder = os.path.exists(path)
    # 判斷結果
    if not folder:
        os.makedirs(path)
        print('------ build success ------')

class NormalDataset(Dataset):
    def __init__(self, target, csv_file = None, transform=None):        

        self.target = torch.from_numpy(target).long()
        self.kmer = torch.from_numpy((pd.read_csv(csv_file,header=None).to_numpy())).float()
        # self.kmer = self.kmer.unsqueeze(1)
        self.transform = transform
        
    def __getitem__(self, index):

        y = self.target[index]      
        x_k = self.kmer[index]
        # if self.transform:
        #     x = self.transform(x)
        x_k = x_k.unsqueeze(0)
        
        return x_k, y
    
    def __len__(self):
        return len(self.kmer)

class Model(nn.Module):
    
    def __init__(self):
        super().__init__()
        #28
        self.conv1 = nn.Conv1d(1,1,kernel_size=3, stride=1)

        self.dropout = nn.Dropout2d(p =0.5)

        self.fc1 = nn.Linear(358, 200)
        self.fc2 = nn.Linear(200, 10)

        
    def forward(self, x):

        print(x.shape)
        print(len(x))
        x = F.relu(self.conv1(x))
        print(x.shape)
        x = x.view(x.shape[0], -1)
        print(x.shape)

        # x = torch.add(x, x2, alpha = 1)
        x = self.dropout(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim = 1)

        return x

# Define train process
def train(model, device, train_loader,loss_func , optimizer, epoch):
    model.train()
    for batch_idx, (kmer_code, target) in enumerate(train_loader):

        kmer_code, target = kmer_code.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(kmer_code)

        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.item()))
 
# Hyper Parameters
EPOCH = 2
Batch_size = 64
LR = 0.001
degree = 10
k = 10

(train_images, train_labels), (test_images, test_labels) = load_data()

images = np.concatenate((train_images, test_images), axis=0)
labels = np.concatenate((train_labels, test_labels), axis=0)

trainDataset = NormalDataset(labels, 'ori_kmer/k10-d10.csv', transform=None)
trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size= Batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = Model().to(device) # Set the model and move to GPU.
optimizer = optim.Adam(net.parameters(), lr=LR)  # Optimizer
loss_func = nn.CrossEntropyLoss()  # loss function

for batch_idx, (kmer, target) in enumerate(trainLoader):

    if batch_idx == 0:
        print(kmer.numpy().shape)
        print(type(kmer))
    else:
        break
#     for epoch in range(1, EPOCH + 1):
#         train(net, device, trainLoader, loss_func, optimizer, epoch)

# for batch_idx, (data, kmer ,target) in enumerate(testLoader):
#     if batch_idx == 0:
#         image_train = data[0].numpy()
#         print(kmer.numpy().shape)
#     else:
#         break

# Net summary
summary(net, (1, 360), batch_size=64 ,device='cuda')
# random_img = torch.Tensor(np.random.random(size=(1, 1, 360))).to(device)
# out = net(random_img)
# print(out.shape)
