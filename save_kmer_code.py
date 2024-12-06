import numpy as np
# import random
# import time
# import math
import csv
# import cv2
import os
# import pandas as pd
# import torch
# import torch.nn as nn
# from torch import optim
# from torchsummary import summary
# import torch.nn.functional as F
# from torchvision import transforms
# from torch.utils.data import Dataset, DataLoader

from get_kmer import Kmer_extractor

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


(train_images, train_labels), (test_images, test_labels) = load_data()

images = np.concatenate((train_images, test_images), axis=0)
labels = np.concatenate((train_labels, test_labels), axis=0)

sequence = True

# for kernel in ['median','gaussian']:

folder = 'ori_kmer_mix' # ori:original kemr code; median: median kernel; gaussian: gaussian kernel
mkdir(folder)

for k in [15]: # k for kmer
    for d in [8]:  # degree
        k = k
        degree = d
        kmer = Kmer_extractor(degree=degree, k=k, kernel='ori', kernel_size=3, sequence = 'mix')
        filename = folder + '/'+ 'k' + str(k) + '-' + 'd' + str(degree) + '.csv'
        print('Start ' + filename)
        with open(filename, 'w', newline='') as csvfile:
            # 建立 CSV 檔寫入器
            writer = csv.writer(csvfile)
            for img in images:                
                kmer_array = kmer.get_kmer_features(img) # comput Kemer               
                writer.writerow(kmer_array)  # Write kmer in csv
        print('Finished ' + filename)

print('Finished all')

# Save kmer to csv
# t1 = time.time()

# t2 = time.time()
# print('time elapsed: ' + str(t2-t1) + ' seconds')

# kmer = get_kmer_array(train_images[0], degree, k).numpy()


# csv_file = 'k10-d10.csv'
# kmer = pd.read_csv(csv_file,header=None).to_numpy()
# kmer_0 = kmer[0]

# trainDataset = NormalDataset(train_images, train_labels,csv_file)
# trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=64, shuffle=False)

# testDataset = MyDataset(test_images, test_labels)
# testLoader = torch.utils.data.DataLoader(testDataset, batch_size = 64, shuffle = True)

# net = Net().to(device) # Set the model and move to GPU.
# optimizer = optim.Adam(net.parameters(), lr=LR)  # Optimizer
# loss_func = nn.CrossEntropyLoss()  # loss function


# Net summary
# summary(net, (1, 28, 28), batch_size=1 ,device='cuda')
# random_img = torch.Tensor(np.random.random(size=(1, 1, 28, 28))).to(device)
# out = net(random_img)
# print(out.shape)

# Kmer time
# t1 = time.time()
# kmer = get_kmer_array(train_images[0], 10,15).numpy()

# print(len(kmer))

# t2 = time.time()
# print('time elapsed: ' + str(t2-t1) + ' seconds')


# for batch_idx, (data, kmer_array, target) in enumerate(trainLoader):
#     if batch_idx == 0:
#         for index,img in enumerate(data):
#             if index == 0:
#                 kmer_a = kmer_array[index].numpy()

