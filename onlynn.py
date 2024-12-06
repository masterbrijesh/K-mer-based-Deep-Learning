import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import cv2
import time
import random
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from network import Model, Net, LeNet, AlexNet
# from get_kmer_func import get_kmer_array
from dataset import CustomFolder, MyDataset ,load_data
from sklearn.model_selection import KFold

from ray import tune

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

# Define train process
def train(model, device, train_loader, loss_func , optimizer, epoch):
    global degree
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # if batch_idx == 0:
        #     print(data[0].numpy())
        #     print(data[0].numpy().shape)
        #     print(target[0].numpy())
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tDegree:{:.0f} '.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.item(), degree))


from sklearn.metrics import precision_score, recall_score, f1_score  # For calculating of precision, recall, f1score

# Define test process
def test(model, device, test_loader):
    # global result
    model.eval()
    correct_total = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    y_true = torch.tensor([], dtype=torch.long, device=device)
    y_pred = torch.tensor([], device=device)
    with torch.no_grad():
        for index , (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            correct = np.squeeze(predicted.eq(target.view_as(predicted)))

            y_true = torch.cat((y_true, target), 0)
            y_pred = torch.cat((y_pred, predicted), 0)
            total += target.size(0)
            correct_total += (predicted == target).sum().item() 

            # Calculate accuracy for each class.
            for i, class_target in enumerate(target):
                class_label = target[i]
                class_correct[class_label] += correct[i].item()
                class_total[class_label] += 1

    print('Accuracy of the network on the 10000 test images: %.2f %%' % (100 * correct_total / total))
    # result.append(round(100 * correct / total,2))
    y_true = y_true.cpu().numpy()  # Transfer type for sklearn evaluating.
    y_pred = y_pred.cpu().numpy()
    
    # All evaluation is based on "weighted" method.
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1socore = f1_score(y_true, y_pred, average='weighted')

    accuracy = round(correct_total / total,3)

    class_accu = 100 * np.array(class_correct) / np.array(class_total)

    return precision, recall ,f1socore, accuracy, class_accu

# Hyper Parameters
EPOCH = 10
k_fold = 10
new_config = {'l1': 120, 'l2': 84, 'lr': 0.0001, 'batch_size': 64}

# Use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
(train_images, train_labels), (test_images, test_labels) = load_data()

# dataset for Kfold
images = np.concatenate((train_images, test_images), axis=0)
labels = np.concatenate((train_labels, test_labels), axis=0)

# Define tranform function 
transform_normalize = transforms.Compose([transforms.Normalize((0.5),(0.5) )])   # Normalization
# transform_rotate = transforms.Compose([transforms.RandomRotation(rotate_degree, expand=False) ])   # Rotate transforms
transform_grayscale = transforms.Compose([transforms.Grayscale(num_output_channels=1)])   #transform to gray scale

# Define CNN optimizer and loss function
# net = LeNet(new_config["l1"], new_config['l2']).to(device) # Set the model and move to GPU.
net = AlexNet().to(device)
optimizer = optim.Adam(net.parameters(), lr=new_config['lr'])  # Optimizer
loss_func = nn.CrossEntropyLoss()  # loss function

precision_list = list()
recall_list = list()
f1socore_list = list()
accuracy_list = list()
class_accu_list = list()

# Load data
(train_images, train_labels), (test_images, test_labels) = load_data()

# dataset for Kfold
images = np.concatenate((train_images, test_images), axis=0)
labels = np.concatenate((train_labels, test_labels), axis=0)    


# 訓練
t1 = time.time()
for degree in [180]:

    # Load data
    transform_rotate = None if degree == 0 else transforms.Compose([transforms.RandomRotation(degree, expand=False)])
    
    # dataset load
    ori_data = MyDataset(images, labels, transform=None)    
    rotate_data = MyDataset(images, labels,transform=transform_rotate)
    
    # Result for each rotated result
    precision_result = list()
    recall_result = list()
    f1socore_result = list()
    accuracy_result = list()
    class_accu_result = list()

    # Kfold split
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=0)

    for i, (train_index, test_index) in enumerate(kf.split(ori_data)):

        train_data = torch.utils.data.Subset(ori_data, train_index)
        test_data = torch.utils.data.Subset(rotate_data, test_index)
        trainLoader = torch.utils.data.DataLoader(train_data, batch_size= new_config['batch_size'], shuffle=True)
        testLoader = torch.utils.data.DataLoader(test_data, batch_size= new_config['batch_size'], shuffle=True)
        
        for epoch in range(1, EPOCH + 1):
            train(net, device, trainLoader, loss_func, optimizer, epoch)        

        # 測試        
        precision, recall , f1socore, accuracy, class_accu  = test(net, device, testLoader)

        precision_result.append(precision)
        recall_result.append(recall)
        f1socore_result.append(f1socore)
        accuracy_result.append(accuracy)
        class_accu_result.append(class_accu)

    precision_list.append(precision_result)
    recall_list.append(recall_result)
    f1socore_list.append(f1socore_result)
    accuracy_list.append(accuracy_result)
    class_accu_list.append(class_accu_result)
        

t4 = time.time()
interval = t4 -t1
minutes = round(interval/60,0)
seconds = round(interval%60,1)
print('Time elapsed: ' + str(minutes) + ' minutes' + str(seconds) + ' seconds')

print('Precision: ' , precision_list)
np.savetxt('precision_LeNet.csv',np.array(precision_list),delimiter=',')
# print('Average precision: ',np.mean(np.array(precision_list)))

print('Recall:', recall_list)
np.savetxt('Recall_LeNet.csv',np.array(recall_list),delimiter=',')
# print('average precision: ',np.mean(np.array(recall_list)))

print('F1score: ', f1socore_list)
np.savetxt('F1score_LeNet.csv',np.array(f1socore_list),delimiter=',')
# print('Average f1score: ',np.mean(np.array(f1socore_list)))

print('Accuracy: ', accuracy_list)
np.savetxt('Accuracy_LeNet.csv',np.array(accuracy_list),delimiter=',')

class_accu_list = np.squeeze(class_accu_list)
print('Class accuracy: ', class_accu_list)
np.savetxt('Class_accuracy_LeNet.csv',np.array(class_accu_list),delimiter=',')

# print('Average accuracy: ',np.mean(np.array(accuracy_list)))
# print(np.mean(np.array(result_list)))

# t2 = time.time() 



# PATH = './nn.pth'
# torch.save(model.state_dict(), PATH)    

# Tensor張量 轉化為 numpy
# a = torch.FloatTensor(2,3)
# print a.numpy();

# numpy 轉化為 Tensor張量
# a = np.ones(5)
# torch.from_numpy(a)

# Train data loader
# trainDataset = MyDataset(train_images, train_labels,transform=None)
# trainDataset = CustomFolder('20size/', IMG_EXTENSIONS, transform=None, target_transform=None)
# trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size= Batch_size, shuffle=True)

# Test data loader
# testDataset = CustomFolder('90degree/', IMG_EXTENSIONS, transform=None, target_transform=None)
# testDataset = MyDataset(test_images, test_labels,transform=None)
# testLoader = torch.utils.data.DataLoader(testDataset, batch_size= Batch_size, shuffle=True)
