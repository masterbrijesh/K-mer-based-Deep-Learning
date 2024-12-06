import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader

import time
import random
import argparse
import numpy as np
import pandas as pd
from PIL import Image

from get_kmer import Kmer_extractor
from network import KmerLeNet, KmerAlexNet
from dataset import load_data, NormalDataset, NormaltestData

from sklearn.model_selection import KFold


# Define train process
def train(model, device, train_loader,loss_func , optimizer, epoch):
    model.train()
    global degree
    for batch_idx, (data, kmer_code, target) in enumerate(train_loader):

        data,kmer_code, target = data.to(device), kmer_code.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data, kmer_code)

        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.item()))
 
from sklearn.metrics import precision_score, recall_score, f1_score  # For calculating of precision, recall, f1score

# Define test process
def test(model, device, test_loader):
    # global result
    model.eval()
    correct = 0
    total = 0
    y_true = torch.tensor([], dtype=torch.long, device=device)
    y_pred = torch.tensor([], device=device)
    with torch.no_grad():
        for batch , (data, kmer_code, target) in enumerate(test_loader):
            data, kmer_code, target = data.to(device), kmer_code.to(device), target.to(device)
            output = model(data, kmer_code)
            _, predicted = torch.max(output.data, 1)   
            y_true = torch.cat((y_true, target), 0)
            y_pred = torch.cat((y_pred, predicted), 0)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    y_true = y_true.cpu().numpy()  # Transfer type for sklearn evaluating.
    y_pred = y_pred.cpu().numpy()
    print('Accuracy of the network on the 10000 test images: %.2f %%' % (100 * correct / total))
    # print(len(y_true))
    accuracy = round(correct / total,2)
    # All evaluation is based on "weighted" method.
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1socore = f1_score(y_true, y_pred, average='weighted')
    # result.append(round(100 * correct / total,2))

    return precision, recall ,f1socore, accuracy



# Hyper Parameters
EPOCH = 10
Batch_size = 64
LR = 0.0001
kmer_degree = 8
kmer_k = 15 
kernel = 'ori'  # ori:original kemr code; median: median kernel; gaussian: gaussian kernel
# rotate_degree = 90
kmer_weight = 0.5
kernel_size = 3
sequence = 'mix'

# Use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define tranform function for training
transform_normalize = transforms.Compose([transforms.Normalize((0.5),(0.5))])

# Load data
(train_images, train_labels), (test_images, test_labels) = load_data()

# Load kmer data
# csv_file = kernel + '_kmer/' + 'k' + str(kmer_k) + '-d' + str(kmer_degree) +'.csv'
if kernel_size == 5:
    if sequence == True:
        csv_file = kernel + '_kmer5/' + 'k' + str(kmer_k) + '-d' + str(kmer_degree) +'.csv' 
    elif sequence == False:
        csv_file = kernel + '_kmer5_unseq/' + 'k' + str(kmer_k) + '-d' + str(kmer_degree) +'.csv'  
    elif sequence == 'mix':
        csv_file = kernel + '_kmer5_mix/' + 'k' + str(kmer_k) + '-d' + str(kmer_degree) +'.csv'
elif kernel_size == 3:
    if sequence == True:
        csv_file = kernel + '_kmer/' + 'k' + str(kmer_k) + '-d' + str(kmer_degree) +'.csv' 
    elif sequence == False:
        csv_file = kernel + '_kmer_unseq/' + 'k' + str(kmer_k) + '-d' + str(kmer_degree) +'.csv'  
    elif sequence == 'mix':
        csv_file = kernel + '_kmer_mix/' + 'k' + str(kmer_k) + '-d' + str(kmer_degree) +'.csv'

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

# dataset for Kfold
images = np.concatenate((train_images, test_images), axis=0)
labels = np.concatenate((train_labels, test_labels), axis=0)
# ori_data = NormalDataset(images, labels, csv_file, transform=None, transform_kmer=None)
# if kernel == 'ori':
#     kernel = None
# rotate_data = NormaltestData(images, labels, degree=kmer_degree, k=kmer_k, kernel=kernel, transform=transform_rotate, transform_kmer=None)

kmer_length = int((360/kmer_degree)* kmer_k) + 400   # LeNet+400; AlexNet+384
net = KmerLeNet(lengh=kmer_length, alpha=kmer_weight).to(device) # Set the model and move to GPU.
optimizer = optim.Adam(net.parameters(), lr=LR)  # Optimizer
loss_func = nn.CrossEntropyLoss()  # loss function

kf = KFold(n_splits=10, shuffle=True, random_state=0)

precision_list = list()
recall_list = list()
f1socore_list = list()
accuracy_list = list()

degree_list = [0, 45, 90, 135]  # 0,45,90,135

t1 = time.time()

    
# Load original dataset for training.
ori_data = NormalDataset(images, labels, csv_file, transform=None, transform_kmer=None)

if kernel == 'ori':
    kernel = None


for i, (train_index, test_index) in enumerate(kf.split(ori_data)):
    train_data = torch.utils.data.Subset(ori_data, train_index)
    # test_data = torch.utils.data.Subset(rotate_data, test_index)
    trainLoader = torch.utils.data.DataLoader(train_data, batch_size= Batch_size, shuffle=True)
    # testLoader = torch.utils.data.DataLoader(test_data, batch_size= Batch_size, shuffle=True)
    
    for epoch in range(1, EPOCH + 1):
        train(net, device, trainLoader, loss_func, optimizer, epoch)

    # Result for each rotated degree result
    precision_result = list()
    recall_result = list()
    f1socore_result = list()
    accuracy_result = list()

    for degree in degree_list:
        # Define transform degree for test data.
        transform_rotate = None if degree == 0 else transforms.Compose([transforms.RandomRotation(degree, expand=False)])

        rotate_data = NormaltestData(images, labels, degree=kmer_degree, k=kmer_k, kernel=kernel, kernel_size=kernel_size, sequence=sequence, transform=transform_rotate, transform_kmer=None)            
        
        test_data = torch.utils.data.Subset(rotate_data, test_index)
        testLoader = torch.utils.data.DataLoader(test_data, batch_size= Batch_size, shuffle=True)
        
        # Testing
        precision, recall , f1socore, accuracy = test(net, device, testLoader)
        precision_result.append(precision)
        recall_result.append(recall)
        f1socore_result.append(f1socore)
        accuracy_result.append(accuracy)

    precision_list.append(precision_result)
    recall_list.append(recall_result)
    f1socore_list.append(f1socore_result)
    accuracy_list.append(accuracy_result)

    # np.savetxt( str(degree) + '_KmerNN.csv',np.array(result),delimiter=',')
        


t4 = time.time()
interval = t4 -t1
minutes = round(interval/60,0)
seconds = round(interval%60,1)
print('Time elapsed: ' + str(minutes) + ' minutes, ' + str(seconds) + ' seconds')

print('Precision: ' , precision_list)
np.savetxt('precision_kmernn.csv',np.array(precision_list),delimiter=',')
# print('Average precision: ',np.mean(np.array(precision_list)))

print('Recall:', recall_list)
np.savetxt('Recall_kmernn.csv',np.array(recall_list),delimiter=',')
# print('average precision: ',np.mean(np.array(recall_list)))

print('F1score: ', f1socore_list)
np.savetxt('F1score_kmernn.csv',np.array(f1socore_list),delimiter=',')
# print('Average f1score: ',np.mean(np.array(f1socore_list)))

print('Accuracy: ', accuracy_list)
np.savetxt('Accuracy_kmernn.csv',np.array(accuracy_list),delimiter=',')
# print('Average accuracy: ',np.mean(np.array(accuracy_list)))




'''
End -------------------------------------------------------------------------------------
'''

# Tensor張量 轉化為 numpy
# a = torch.FloatTensor(2,3)
# print a.numpy();

# numpy 轉化為 Tensor張量
# a = np.ones(5)
# torch.from_numpy(a)


# Train data loader
# trainDataset = NormalDataset(train_images, train_labels, csv_file, transform=None)
# trainDataset = RotateDataset('20size/', IMG_EXTENSIONS, degree=degree, k=k, transform=None, target_transform=None)
# trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size= Batch_size, shuffle=True)

# Test data loader
# testDataset = RotateDataset('90degree/', IMG_EXTENSIONS, degree=degree, k=k, transform=None, target_transform=None)
# testDataset = NormaltestData(test_images, test_labels, degree=degree, k=k, transform=None)
# testLoader = torch.utils.data.DataLoader(testDataset, batch_size= Batch_size, shuffle=True)


'''
# 5 layers Network structure
class Net(nn.Module):
    
    def __init__(self):
        super().__init__()
        #28
        self.conv1 = nn.Conv2d(1, 16, kernel_size = 3, stride = 1, padding = 1)
        #28 
        #14
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding = 1)
        #14
        #7
        self.conv3 = nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 0)
        #5
        #3 pool2
        self.conv4 = nn.Conv2d(64, 96, kernel_size = 3, stride = 1, padding = 1)
        #3
        self.conv5 = nn.Conv2d(96, 96, kernel_size = 2, stride = 1, padding = 0)
        #2        
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
        self.pool2 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        # kemr: 360; nn: 384; total=852
        self.fc1 = nn.Linear(384, 270)
        self.fc2 = nn.Linear(270, 200)
        self.fc3 = nn.Linear(200, 100)
        self.fc4 = nn.Linear(100, 10)
        self.dropout = nn.Dropout(p = 0.2)
        
    def forward(self, x, kmer_array):
        
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(x.shape[0], -1)
        # x2 = kmer_array
        x2 = F.pad(kmer_array,(0,24))
        # x = torch.cat((x, x2), dim=1)
        x = torch.add(x, x2, alpha = 0.1)
        x = self.dropout( F.relu(self.fc1(x)) )        
        x = self.dropout( F.relu(self.fc2(x)) )
        # x = torch.cat((x, x2), dim=1)
        x = self.dropout( F.relu(self.fc3(x)) )
        x = F.log_softmax(self.fc4(x), dim = 1)
        
        return x

# 2 layers
class Model(nn.Module):
    
    def __init__(self):
        super().__init__()
        #28
        self.conv1 = nn.Conv2d(1, 16, kernel_size = 5, stride = 1, padding = 1)

        self.conv2 = nn.Conv2d(16, 32, kernel_size = 5, stride = 1, padding = 1)

        self.conv3 = nn.Conv1d(1,1,kernel_size=3)

        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)

        self.dropout = nn.Dropout2d(p =0.5)

        self.fc1 = nn.Linear(800, 200) 
        self.fc2 = nn.Linear(200, 10)

        
    def forward(self, x, kmer_array):
        # print(x.shape)
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool1(self.conv2(x)))
        # x = get_kmer_array(x)
        
        x = x.view(x.size(0), -1)
        x2 = kmer_array
        # x2 = F.pad(kmer_array,(0,24))
        # x = torch.cat((x, x2), dim=1)
        x2 = F.relu(self.conv3(x2))
        x2 = F.pad(kmer_array,(0,440))
        # print(x.shape)
        print(x2.shape)
        x = torch.add(x, x2, alpha = 1)
        x = self.dropout(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim = 1)

        return x



for i in range(5):
    # 訓練
    t1 = time.time()
    for epoch in range(1, EPOCH + 1):
        train(net, device, trainLoader, loss_func, optimizer, epoch)
        print('Epoch ' + str(epoch) +' done.')
    t2 = time.time()
    print('Finished Training. Training time elapsed: ' + str(t2-t1) + ' seconds')

    # 測試
    t3 = time.time()
    test(net, device, testLoader)
    t4 = time.time()
    print('Finished Testing. Testing time elapsed: ' + str(t4-t3) + ' seconds')
'''

