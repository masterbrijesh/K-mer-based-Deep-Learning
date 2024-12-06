
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader

import os
import cv2
import csv
import time
import random
import argparse
import numpy as np

from network import Model, Net, LeNet, KmerDensNet
from dataset import RotateDataset, load_data, NormalDataset, NormaltestData

from functools import partial
from sklearn.model_selection import KFold

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import tensorboard

# Define train process
def train(model, device, train_loader, loss_func , optimizer, epoch):

    # global degree
    model.train()
    for batch_idx, (data, kmer_code, target) in enumerate(train_loader):

        data, kmer_code, target = data.to(device), kmer_code.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data, kmer_code)                  
        
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()


# Define test process
def test(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for index , (data, kmer_code, target) in enumerate(test_loader):
            data, kmer_code, target = data.to(device), kmer_code.to(device), target.to(device)
            output = model(kmer_code)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    # print(total)
    print('Accuracy of the network on the 10000 test images: %.2f %%' % (100 * correct / total))
    return correct / total

# Hyper Parameters
EPOCH = 10


def get_data_loaders(config):
    # Load data
    (train_images, train_labels), (test_images, test_labels) = load_data()

    # dataset for all images and labels connnect all the data
    images = np.concatenate((train_images, test_images), axis=0)
    labels = np.concatenate((train_labels, test_labels), axis=0)    
    
    # Split train and test index
    train_index = list(range(len(labels))) 
    test_index = random.sample(range(len(labels)), k=int(len(labels)*0.1))
    for element in test_index:
        train_index.remove(element)
    
    # Transform function of rotating image in 45 degree.
    transform_rotate = transforms.Compose([transforms.RandomRotation(45, expand=False)])

    # Load kmer data from csv
    file_dir = 'C:/Users/dal/Desktop/code/'
    csv_file = file_dir + config['kernel'] + '_kmer/' + 'k' + str(config['kmer_k']) + '-d' + str(config['kmer_degree']) +'.csv'

    # Dataset load from own defined class 
    ori_data = NormalDataset(images, labels, csv_file, transform=None, transform_kmer=None)    
    rotate_data = NormaltestData(images, labels, degree=config['kmer_degree'], k=config['kmer_k'], kernel=config['kernel'], 
            kernel_size=3, transform=transform_rotate, transform_kmer=None)    
    
    # Split data with index
    train_data = torch.utils.data.Subset(ori_data, train_index)
    test_data = torch.utils.data.Subset(rotate_data, test_index)

    trainLoader = torch.utils.data.DataLoader(train_data, batch_size= config['batch_size'], shuffle=True)
    testLoader = torch.utils.data.DataLoader(test_data, batch_size= config['batch_size'], shuffle=True)

    return trainLoader, testLoader


def train_mnist(config):
    global EPOCH
    # Use GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load data
    train_loader, test_loader = get_data_loaders(config)
    # Load model
    kmer_length = int((360/config['kmer_degree'])* config['kmer_k'])
    net = KmerDensNet(lengh=kmer_length).to(device)
    optimizer = optim.Adam(net.parameters(), lr=config["lr"])  # Optimizer
    loss_func = nn.CrossEntropyLoss()  # loss function

    train(net, device , train_loader, loss_func, optimizer,  EPOCH)
    # accuracy = test(net, device ,test_loader)

    # Validation loss
    val_loss = 0.0
    val_steps = 0
    total = 0
    correct = 0
    for i, (inputs, kmer_code, labels) in enumerate(test_loader):
        with torch.no_grad():

            inputs, kmer_code, labels = inputs.to(device), kmer_code.to(device), labels.to(device)
            outputs = net(kmer_code)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = loss_func(outputs, labels)
            val_loss += loss.cpu().numpy()
            val_steps += 1

    # with tune.checkpoint_dir(epoch) as checkpoint_dir:
    #     path = os.path.join(checkpoint_dir, "checkpoint")
    #     torch.save((net.state_dict(), optimizer.state_dict()), path)

    tune.report(loss=(val_loss / val_steps), mean_accuracy=correct / total)
    print("Finished Training")


num_samples = 4560
max_num_epochs = 10 
gpus_per_trial = 0.1

config = {
    # "l1": tune.sample_from(lambda _: 2 ** np.random.randint(6, 9)),
    # "l2": tune.sample_from(lambda _: 2 ** np.random.randint(6, 9)),
    "lr": tune.quniform(0.01, 0.0005,0.0005),
    "batch_size": tune.choice([32, 64, 128, 256, 512]),
    'kmer_k':tune.choice([8, 10, 12, 15]),
    'kmer_degree':tune.choice([5, 8, 10, 15]),     
    'kernel':tune.choice(['ori','median','gaussian'])
    }
scheduler = ASHAScheduler(
    metric="loss",
    mode="min",
    max_t=max_num_epochs,
    grace_period=1,
    reduction_factor=2)
reporter = CLIReporter(
    # parameter_columns=["l1", "l2", "lr", "batch_size"],
    metric_columns=["loss", "accuracy", "training_iteration"])

t1 = time.time()

analysis = tune.run(
    train_mnist,
    resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
    config=config,
    num_samples=num_samples,
    scheduler=scheduler,
    progress_reporter=reporter,
    fail_fast = True)   # To stop the entire Tune run as soon as any trail errors.

t4 = time.time()
interval = t4 -t1
minutes = round(interval/60, 0)
seconds = round(interval%60, 1)

print('Time elapsed: ' + str(minutes) + ' minutes' + str(seconds) + ' seconds')
print("Best config: ", analysis.get_best_config(metric="loss",mode='min'))

best_config = analysis.get_best_config(metric="loss",mode='min')

best_config['optimize_time'] = str(minutes) + ' minutes' + str(seconds) + ' seconds'

with open('Best_config.csv', 'w') as f:
    for key in best_config.keys():
        f.write("%s,%s\n"%(key, best_config[key]))


# k-fold validated with best config --------------------------------------------------------

# Define train process
def train_best_config(model, device, train_loader, loss_func , optimizer, epoch):
    global degree
    model.train()
    for batch_idx, (data, kmer_code, target) in enumerate(train_loader):

        data, kmer_code, target = data.to(device), kmer_code.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(kmer_code)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tDegree:{:.0f} '.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.item(), degree))

# Define test process
def test_best_config(model, device, test_loader):
    global result
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for index , (data, kmer_code, target) in enumerate(test_loader):
            data, kmer_code, target = data.to(device), kmer_code.to(device), target.to(device)
            output = model(kmer_code)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print('Accuracy of the network on the 10000 test images: %.2f %%' % (100 * correct / total))
    result.append(round(100 * correct / total,2))

# Hyper Parameters
EPOCH = 10
k_fold = 10

# Use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
(train_images, train_labels), (test_images, test_labels) = load_data()

# dataset for Kfold
images = np.concatenate((train_images, test_images), axis=0)
labels = np.concatenate((train_labels, test_labels), axis=0)

# Define CNN optimizer and loss function
kmer_length = int((360/best_config['kmer_degree'])* best_config['kmer_k'])
net = KmerDensNet(lengh=kmer_length).to(device)  # Set the model and move to GPU.
optimizer = optim.Adam(net.parameters(), lr=best_config['lr'])  # Optimizer
loss_func = nn.CrossEntropyLoss()  # loss function

result = list()
result_list = list()

# Load data
(train_images, train_labels), (test_images, test_labels) = load_data()

# dataset for Kfold
images = np.concatenate((train_images, test_images), axis=0)
labels = np.concatenate((train_labels, test_labels), axis=0)


# CSV file base on best config
csv_file = best_config['kernel'] + '_kmer/' + 'k' + str(best_config['kmer_k']) + '-d' + str(best_config['kmer_degree']) +'.csv'

# 訓練
t5 = time.time()
for degree in [0,45,90,135]:

    # Load data
    transform_rotate = None if degree == 0 else transforms.Compose([transforms.RandomRotation(degree, expand=False)])
    
    # dataset load
    ori_data = NormalDataset(images, labels, csv_file, transform=None, transform_kmer=None)    
    rotate_data = NormaltestData(images, labels, degree=best_config['kmer_degree'], k=best_config['kmer_k'], kernel=best_config['kernel'], 
            kernel_size=3, transform=transform_rotate, transform_kmer=None)    
    
    # Result for each rotated result
    result = list()

    # Kfold split
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=0)
    t2 = time.time()
    for i, (train_index, test_index) in enumerate(kf.split(ori_data)):

        train_data = torch.utils.data.Subset(ori_data, train_index)
        test_data = torch.utils.data.Subset(rotate_data, test_index)
        trainLoader = torch.utils.data.DataLoader(train_data, batch_size= best_config['batch_size'], shuffle=True)
        testLoader = torch.utils.data.DataLoader(test_data, batch_size= best_config['batch_size'], shuffle=True)
        
        for epoch in range(1, EPOCH + 1):
            train_best_config(net, device, trainLoader, loss_func, optimizer, epoch)        

        # 測試        
        test_best_config(net, device, testLoader)
        
        # print('Finished Testing. Testing time elapsed: ' + str(t4-t3) + ' seconds')
    t3 = time.time()
    print('Finished Training. Training time elapsed: ' + str(t3-t2) + ' seconds')    
    result_list.append(result)
    np.savetxt( str(degree) + '_KmerFC.csv',np.array(result),delimiter=',')

np.savetxt('result__KmerFC.csv',np.array(result_list),delimiter=',')

t6 = time.time()
interval = t6 -t5
minutes = round(interval/60,0)
seconds = round(interval%60,1)
best_config['training_time'] = str(minutes) + ' minutes' + str(seconds) + ' seconds'

with open('Best_config.csv', 'w') as f:
    for key in best_config.keys():
        f.write("%s,%s\n"%(key, best_config[key]))

print("Best config: ", analysis.get_best_config(metric="loss",mode='min'))
print('Time elapsed: ' + str(minutes) + ' minutes' + str(seconds) + ' seconds')
print(result_list)






















# tensorboard --logdir ~/ray_result

'''
def train_mnist(config, checkpoint_dir=None):
    net = AlexNet(config["l1"], config["l2"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=config["lr"])  # Optimizer
    criterion = nn.CrossEntropyLoss()  # loss function

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    trainset, testset = get_data_loaders()

    # test_abs = int(len(trainset) * 0.8)
    # train_subset, val_subset = random_split(
    #     trainset, [test_abs, len(trainset) - test_abs])

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8)

    valloader = torch.utils.data.DataLoader(
        testset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8)

    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, (inputs,labels) in enumerate(valloader):
            with torch.no_grad():

                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((net.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps), accuracy=correct / total)
    print("Finished Training")

def main(num_samples=10, max_num_epochs=10, gpus_per_trial=0.1):

    config = {
        "l1": tune.sample_from(lambda _: 2 ** np.random.randint(5, 9)),
        "l2": tune.sample_from(lambda _: 2 ** np.random.randint(4, 9)),
        "lr": tune.loguniform(1e-5, 1e-1),
        "batch_size": tune.choice([32, 64, 128, 256, 512])
    }
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    result = tune.run(
        tune.with_parameters(train_mnist),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        metric="loss",
        mode="min",
        num_samples=num_samples,
        scheduler=scheduler
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    # np.savetxt('best_trial_nn.csv',np.array(best_trial),delimiter=',')
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))
    # np.savetxt('best_trial_nn.csv',np.array(best_trial),delimiter=',')
    # best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    # best_trained_model.to(device)

    checkpoint_path = os.path.join(best_trial.checkpoint.value, "checkpoint")

    model_state, optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)

    # t2 = time.time()
    # # Load data
    # (train_images, train_labels), (test_images, test_labels) = load_data()

    # # dataset for Kfold
    # images = np.concatenate((train_images, test_images), axis=0)
    # labels = np.concatenate((train_labels, test_labels), axis=0)

    # # Define CNN optimizer and loss function
    # net = AlexNet(best_trial.config["l1"], best_trial.config['l2']).to(device) # Set the model and move to GPU.
    # optimizer = optim.Adam(net.parameters(), lr=best_trial.config['lr'])  # Optimizer
    # loss_func = nn.CrossEntropyLoss()  # loss function
    # result_list = list()
    # for degree in [0,45,90,135]:
    
    #     # Load data
    #     transform_rotate = None if degree == 0 else transforms.Compose([transforms.RandomRotation(degree, expand=False)])
        
    #     # dataset load
    #     ori_data = MyDataset(images, labels, transform=None)    
    #     rotate_data = MyDataset(images, labels,transform=transform_rotate)
        
    #     # Result for each rotated result
    #     result = list()

    #     # Kfold split
    #     kf = KFold(n_splits=10, shuffle=True, random_state=0)

    #     for i, (train_index, test_index) in enumerate(kf.split(ori_data)):

    #         train_data = torch.utils.data.Subset(ori_data, train_index)
    #         test_data = torch.utils.data.Subset(rotate_data, test_index)
    #         trainLoader = torch.utils.data.DataLoader(train_data, batch_size= new_config['batch_size'], shuffle=True)
    #         testLoader = torch.utils.data.DataLoader(test_data, batch_size= new_config['batch_size'], shuffle=True)
            
    #         for epoch in range(1, EPOCH + 1):
    #             train(net, device, trainLoader, loss_func, optimizer, epoch)        

    #         # 測試        
    #         test(net, device, testLoader)
        
        
    #     # print('Finished Training. Training time elapsed: ' + str(t3-t2) + ' seconds')    
    #     result_list.append(result)
    #     np.savetxt( str(degree) + '_nn.csv',np.array(result),delimiter=',')
    #     # print("Best trial test set accuracy: {}".format(test_acc))
    # t3 = time.time()
    # np.savetxt('result_nn.csv',np.array(result_list),delimiter=',')
    # print("Best trial config: {}".format(best_trial.config))
    # print("Best trial final validation loss: {}".format(
    #     best_trial.last_result["loss"]))
    # print("Best trial final validation accuracy: {}".format(
    #     best_trial.last_result["accuracy"]))

if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=30, max_num_epochs=10, gpus_per_trial=0.1)
'''

# interval = t3 - t2
# minutes = round(interval/60,0)
# seconds = round(interval%60,1)
# print('Time elapsed: ' + str(minutes) + ' minutes' + str(seconds) + ' seconds')

# print(result_list)


# np.savetxt('train.csv',test_data,delimiter=',')