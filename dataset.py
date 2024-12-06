import numpy as np
import cv2
import os
import random
import torch
import torch.nn as nn
import torchvision
from torch import optim
# from torchsummary import summary
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
# from torchvision.datasets.folder import default_loader
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

from get_kmer import Kmer_extractor


def load_data(path= "C:/Users/dal/Desktop/code/MNIST_data/mnist.npz"):
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)

def pil_loader(path):    # 一般採用pil_loader函式。
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')  # Convert to grayscale

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def find_classes(dir): 
    # classes is the name of all the folders
    # class_to_idx give the name for.
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    # 将文件的名变成小写
    filename_lower = filename.lower()

    # endswith() 方法用于判断字符串是否以指定后缀结尾
    # 如果以指定后缀结尾返回True，否则返回False
    return any(filename_lower.endswith(ext) for ext in extensions)

def make_dataset(dir, class_to_idx, extensions):  #dir=r'D:\PycharmProjects\DenseNet\image\raw_img\train'
    images = []
    dir = os.path.expanduser(dir)  #D:\PycharmProjects\DenseNet\image\raw_img\train
    for target in sorted(os.listdir(dir)):  #target=black break hide shade sna五种类别
        d = os.path.join(dir, target)    #此时d输出包含五种类别的五个绝对路径
        if not os.path.isdir(d):   #此句在判断d是否是一个路径
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):  # 文件的后缀名是否符合给定
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname) #该路径已经具体到每张图片
                    item = (path, class_to_idx[target]) #将每张图片与对应的标签放到一个元组
                    images.append(item) #将每张图片与对应的标签形成的元组，加入到image列表中

    return images

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)),cmap='gray')
    plt.show()

# Image from folder with no kmer code
class CustomFolder(Dataset):

    def __init__(self, root, extensions, transform=None, target_transform=None,
                loader=default_loader):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx, extensions)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(extensions)))
        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        index (int): Index
	    Returns:tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = torch.from_numpy(np.array(self.loader(path))).float()
        img = img.unsqueeze(0)
        # img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imgs)

# kmer code from csv file
class NormalDataset(Dataset):
    def __init__(self, data, target, csv_file = None, transform=None , transform_kmer=None):        
        # Initialize paths, transforms, and so on
        self.data = torch.from_numpy(data).float()
        self.data = self.data.unsqueeze(1)
        self.target = torch.from_numpy(target).long()
        self.kmer = torch.from_numpy((pd.read_csv(csv_file,header=None).to_numpy())).float()
        # self.kmer = self.kmer.unsqueeze(1)
        self.transform = transform
        self.transform_kmer = transform_kmer

        
    def __getitem__(self, index):
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        x = self.data[index]
        y = self.target[index]      
        x_k = self.kmer[index]

        if self.transform:
            x = self.transform(x) 
        
        if self.transform_kmer:
            x_k = self.transform_kmer(x_k) 
                   
        x_k = x_k.unsqueeze(0)
        
        return x, x_k, y
    
    def __len__(self):
        # Indicate the total size of the dataset
        return len(self.data)

# data and target.
class MyDataset(Dataset):
    def __init__(self, data, target, transform=None):        
        # Initialize paths, transforms, and so on
        self.data = torch.from_numpy(data).float()
        #print(self.data.shape)
        self.data = self.data.unsqueeze(1)
        #print("Now",self.data.shape)
        self.target = torch.from_numpy(target).long()
        self.transform = transform
        
    def __getitem__(self, index):
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        x = self.data[index]
        y = self.target[index]
        # x_k = get_kmer_array(self.data[index][0],10,13)

        if self.transform:
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        # Indicate the total size of the dataset
        return len(self.data)

# Image from folder and calculatle kmer code.
class FolderWithKmer(Dataset):
    def __init__(self, root, extensions,degree = 8, k =15, kernel=None, kernel_size = 3, sequence=True, transform=None, target_transform=None,
                loader=default_loader):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx, extensions)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(extensions)))
        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        # self.sequence = sequence
        # self.degree = degree
        # self.length = k
        self.kmer = Kmer_extractor(degree, k, kernel, kernel_size, sequence)

    def __getitem__(self, index):
        """
        index (int): Index
	    Returns:tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]

        img = np.array(self.loader(path))

        # x_k = torch.from_numpy(self.kmer.get_kmer_features(img)).float()
        img = torch.from_numpy(img).float()
        img = img.unsqueeze(0)                
        # x_k = x_k.unsqueeze(0)

        target = target

        if self.transform is not None:
            img = self.transform(img)

        x_k = torch.from_numpy(self.kmer.get_kmer_features(img.numpy()[0])).float()
        x_k = x_k.unsqueeze(0)
        
        if self.target_transform is not None:
            target = self.target_transform(target)

        
        return img, x_k, target

    def __len__(self):
        return len(self.imgs)

# From data and calculate kmer 
class NormaltestData(Dataset):
    def __init__(self, data, target, degree=10, k=10, kernel=None, kernel_size=3, sequence=True, transform=None, transform_kmer=None):        
        # Initialize paths, transforms, and so on
        self.data = torch.from_numpy(data).float()
        self.data = self.data.unsqueeze(1)
        self.target = torch.from_numpy(target).long()
        self.kmer = Kmer_extractor(degree, k, kernel, kernel_size, sequence)
        self.transform = transform
        self.transform_kmer = transform_kmer

        
    def __getitem__(self, index):
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        x = self.data[index]
        y = self.target[index]      
        
        if self.transform:
            x = self.transform(x)

        x_k = torch.from_numpy(self.kmer.get_kmer_features(x.numpy()[0])).float()
        x_k = x_k.unsqueeze(0)

        if self.transform_kmer:
            x_k = self.transform_kmer(x_k)

        return x, x_k, y
    
    def __len__(self):
        # Indicate the total size of the dataset
        return len(self.data)

 
# IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

# # Load data
# (train_images, train_labels), (test_images, test_labels) = load_data()

# # Train data loader
# trainDataset = MyDataset(train_images, train_labels,transform=None)
# trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size= 4, shuffle=True)

# transform_grayscale = transforms.Compose([transforms.Grayscale(num_output_channels=1)]) #,transforms.ToTensor()

# testDataset = RotateDataset('test/', IMG_EXTENSIONS,degree=36, k=10,  transform=None, target_transform=None)
# testLoader = torch.utils.data.DataLoader(testDataset, batch_size= 4, shuffle=True)

# # for batch_idx, (data, target) in enumerate(testLoader):
# #     if batch_idx == 0:
# #         image_test = data[0].numpy()
# #     else:
# #         break

# for batch_idx, (data, kmer ,target) in enumerate(testLoader):
#     if batch_idx == 0:
#         image_train = data[0].numpy()
#         print(kmer.numpy().shape)
#     else:
#         break

# dataiter = iter(trainLoader)
# images, labels = dataiter.next()

# # print images
# classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
# imshow(torchvision.utils.make_grid(images))
# print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))