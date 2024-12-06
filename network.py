import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import numpy as np


# N = (W-F+2P)/S+1 (2 layers)
class Model(nn.Module):
    
    def __init__(self):
        super().__init__()
        #28
        self.conv1 = nn.Conv2d(1, 16, kernel_size = 5, stride = 1, padding = 1)

        self.conv2 = nn.Conv2d(16, 32, kernel_size = 5, stride = 1, padding = 1)

        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)

        self.dropout = nn.Dropout2d(p =0.5)

        self.fc1 = nn.Linear(800, 200)
        self.fc2 = nn.Linear(200, 10)

        
    def forward(self, x):
        
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool1(self.conv2(x)))
        # x = get_kmer_array(x)
        
        x = x.view(x.size(0), -1)
        x = self.dropout(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim = 1)

        return x

# N = (W-F+2P)/S+1 (5 layers)
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
        # kemr: 468; nn: 384; total=852
        self.fc1 = nn.Linear(384, 270)
        self.fc2 = nn.Linear(270, 200)
        self.fc3 = nn.Linear(200, 100)
        self.fc4 = nn.Linear(100, 10)
        self.dropout = nn.Dropout(p = 0.2)
        
    def forward(self, x):
        
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
        # x = torch.cat((x, x2), dim=1)
        x = self.dropout( F.relu(self.fc1(x)) )
        x = self.dropout( F.relu(self.fc2(x)) )
        x = self.dropout( F.relu(self.fc3(x)) )
        x = F.log_softmax(self.fc4(x), dim = 1)
        
        return x

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        # x = x.view(x.shape[0], -1)
        # x = x.view(-1, 16*5*5)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class AlexNet(nn.Module):
    def __init__(self,l1=270, l2=200):
        super(AlexNet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dense1 = nn.Linear(384,l1)
        self.drop1 = nn.Dropout(0.2)
        self.dense2 = nn.Linear(l1,l2)
        self.drop2 = nn.Dropout(0.2)
        self.dense3 = nn.Linear(l2,10)

    def forward(self,x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv5(F.relu(self.conv4(F.relu(self.conv3(x)))))))
        x = torch.flatten(x, 1)
        x = self.dense3(self.drop2(F.relu(self.dense2(self.drop1(F.relu(self.dense1(x)))))))
        # x = F.log_softmax(x, dim = 1)
        return x

# LeNet + Kmer
class KmerLeNet(nn.Module):
    def __init__(self,lengh=400, alpha=1):
        super(KmerLeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        self.dropout = nn.Dropout2d(p =0.5)

        if lengh>400:
            self.fc1 = nn.Linear(lengh, 512)
        else:
            self.fc1 = nn.Linear(400, 120)

        # self.fc1 = nn.Linear(in_features=400 + lengh, out_features=120)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=10)
        self.alpha = alpha


    def forward(self, x, kmer_array):
        x = F.relu(self.conv1(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = torch.flatten(x, 1)
        # print(x)

        # if len(x[1])>= len(kmer_array[1]):
        #     x2 = F.pad(kmer_array,(0,len(x[1])-len(kmer_array[1])))
        # else:
        #     x2 = kmer_array
        #     x = F.pad(x,(0,len(kmer_array[1])-len(x[1])))
        # x = torch.add(x, x2, alpha = 1)

        # x2 = kmer_array
        # x2 = torch.flatten(x2, 1)
        # x = torch.cat((x, x2), dim=1)
            
        x2 = kmer_array
        # x2 = torch.flatten(x2, 1)

        # if len(x[1])>= len(x2[1]):
        #     x2 = F.pad(x2,(0,len(x[1])-len(x2[1])))
        # else:
        #     x = F.pad(x,(0,len(x2[1])-len(x[1])))

        # x = torch.add(x, x2, alpha = self.alpha)
        x2 = torch.flatten(x2, 1)
        x = torch.cat((x, x2), dim=1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = self.dropout(self.fc1(x))
        # x = self.fc2(x)
        x = self.fc3(x)
        # x = F.log_softmax(x, dim = 1)
        return x

# AlexNet + Kmer
class KmerAlexNet(nn.Module):
    def __init__(self, lengh=384, alpha=1):
        super(KmerAlexNet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        if lengh>384:
            self.dense1 = nn.Linear(lengh, 512)
        else:
            self.dense1 = nn.Linear(384, 512)

        self.drop1 = nn.Dropout(0.2)
        self.dense2 = nn.Linear(512,256)
        self.drop2 = nn.Dropout(0.2)
        self.dense3 = nn.Linear(256,10)

        self.conv99 = nn.Conv1d(1,1,kernel_size=3)
        self.alpha = alpha

    def forward(self,x, kmer_array):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv5(F.relu(self.conv4(F.relu(self.conv3(x)))))))
        x = torch.flatten(x, 1)

        # x2 = F.relu(self.conv99(kmer_array))
        # x2 = torch.flatten(x2, 1)
        # print(len(x[1]))
        # print(len(x2[1]))

        x2 = kmer_array  # Call K-mer code

        # Add code
        # x2 = torch.flatten(x2, 1)
        # if len(x[1])>= len(x2[1]):
        #     x2 = F.pad(x2,(0,len(x[1])-len(x2[1])))
        # else:
        #     x = F.pad(x,(0,len(x2[1])-len(x[1])))
        # x = torch.add(x, x2, alpha = self.alpha)

        # Concate    
        x2 = torch.flatten(x2, 1)
        x = torch.cat((x, x2), dim=1)

        x = self.dense3(self.drop2(F.relu(self.dense2(self.drop1(F.relu(self.dense1(x)))))))
        # x = F.log_softmax(x, dim = 1)
        return x

# Kmer + FC layers
class KmerDensNet(nn.Module):
    def __init__(self,lengh=384):
        super(KmerDensNet,self).__init__()
        # self.conv1 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=1)
        # self.pool1 = nn.AvgPool1d(kernel_size=2,stride=1)
        # self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,padding=1)
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv3 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        # self.conv4 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, padding=1)
        # self.conv5 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1)
        # self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        # self.dense1 = nn.Linear(math.floor((lengh-2)/1)+1,270)
        self.dense1 = nn.Linear(lengh,270)
        self.drop1 = nn.Dropout(0.2)
        self.dense2 = nn.Linear(270,200)
        self.drop2 = nn.Dropout(0.2)
        self.dense3 = nn.Linear(200,10)

        self.conv99 = nn.Conv1d(1,1,kernel_size=3)
        # self.alpha = alpha

    def forward(self, kmer_array):
        # x = self.pool1(F.relu(self.conv1(x)))
        # x = self.pool2(F.relu(self.conv2(x)))
        # x = self.pool3(F.relu(self.conv5(F.relu(self.conv4(F.relu(self.conv3(x)))))))
        
        x = kmer_array
        # x = self.pool1(x)
        x = torch.flatten(x, 1)
        # x2 = torch.flatten(x2, 1)
        # print(len(x[1]))
        # print(len(x2[1]))

        # if len(x[1])>= len(x2[1]):
        #     x2 = F.pad(x2,(0,len(x[1])-len(x2[1])))
        # else:
        #     x = F.pad(x,(0,len(x2[1])-len(x[1])))

        # x = torch.add(x, x2, alpha = self.alpha)

        x = self.dense3(self.drop2(F.relu(self.dense2(self.drop1(F.relu(self.dense1(x)))))))
        # x = F.log_softmax(x, dim = 1)
        return x

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self,inchannel,outchannel,stride=1,shortcut=None):
        super(ResidualBlock,self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel,outchannel,3,stride,1,bias=False),
            nn.BatchNorm2d(outchannel),
            nn.Relu(inplace=True),
            nn.Conv2d(outchannel,outchannel,3,1,1,bias=False),
            nn.BatchNorm2d(outchannel)
        )

        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)

# ResNet
class ResNet(nn.Module):
    def __init__(self, num_class=1000):
        super(ResNet,self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3,64,7,2,3,bias=False),
            nn.BatchNorm2d(64),
            nn.Relu(inplace=True),
            nn.MaxPol2d(3,2,1)
        )
        self.layer1 = self._make_layer(64,64,3)
        self.layer2 = self._make_layer(64,128,4,stride=2)
        self.layer3 = self._make_layer(128,256,6,stride=2)
        self.layer4 = self._make_layer(256,512,3,stride=2)

        self.fc = nn.Linear(512,num_class)
    
    def _make_layer(self,inchannel,outchannel,block_num,stride=1):
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel)
        )        

        layers = []
        layers.append(ResidualBlock(inchannel, outchannel,stride,shortcut))

        for i in range(1,block_num):
            layers.append(ResidualBlock(inchannel, outchannel))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        X = F.avg_pool2d(x,7)
        x = x.view(x.size(0),-1)

        return self.fc(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = AlexNet().to(device) # Set the model and move to GPU.

# Net summary
summary(net, (1, 28, 28), batch_size=1 ,device='cuda')
random_img = torch.Tensor(np.random.random(size=(1, 1, 28, 28))).to(device)
out = net(random_img)
print(out.shape)