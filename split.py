from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import math
import cv2
import os
from scipy import misc

def load_data(path="MNIST_data/mnist.npz"):
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)

def mkdir(path):
    #判斷目錄是否存在
    #存在：True
    #不存在：False
    folder = os.path.exists(path)

    #判斷結果
    if not folder:
        #如果不存在，則建立新目錄
        os.makedirs(path)
        print('----- Success build -----')

def random_rotate(img, degree):
    rows,cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),degree,1) #第三个参数：变换后的图像大小 
    img = cv2.warpAffine(img,M,(rows,cols))    
    return img

(train_images, train_labels), (test_images, test_labels) = load_data()

from PIL import Image
from matplotlib import cm

from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2

# csv_file = 'k8-d10.csv'
# kmer_code = pd.read_csv(csv_file,header=None).to_numpy()

(train_images, train_labels), (test_images, test_labels) = load_data()

images = np.concatenate((train_images, test_images), axis=0)
labels = np.concatenate((train_labels, test_labels), axis=0)

def balanced_sample_maker(X, y, sample_size, random_seed=42):
    uniq_levels = np.unique(y)
    uniq_counts = {level: sum(y == level) for level in uniq_levels}

    if not random_seed is None:
        np.random.seed(random_seed)

    # find observation index of each class levels
    groupby_levels = {}
    for ii, level in enumerate(uniq_levels):
        obs_idx = [idx for idx, val in enumerate(y) if val == level]
        groupby_levels[level] = obs_idx
    # oversampling on observations of each label
    balanced_copy_idx = []
    for gb_level, gb_idx in groupby_levels.items():
        over_sample_idx = np.random.choice(gb_idx, size=sample_size, replace=True).tolist()
        balanced_copy_idx += over_sample_idx
    np.random.shuffle(balanced_copy_idx)

    data_train = X[balanced_copy_idx]
    labels_train = y[balanced_copy_idx]
    if  ((len(data_train)) == (sample_size*len(uniq_levels))):
        print('number of sampled example ', sample_size*len(uniq_levels), 'number of sample per class ', sample_size, ' #classes: ', len(list(set(uniq_levels))))
    else:
        print('number of samples is wrong ')

    labels, values = zip(*Counter(labels_train).items())
    print('number of classes ', len(list(set(labels_train))))
    check = all(x == values[0] for x in values)
    print(check)
    if check == True:
        print('Good all classes have the same number of examples')
    else:
        print('Repeat again your sampling your classes are not balanced')
    indexes = np.arange(len(labels))
    width = 0.5
    plt.bar(indexes, values, width)
    plt.xticks(indexes + width * 0.5, labels)
    plt.show()
    return data_train,labels_train

sample_size = 100
X_test, y_test = balanced_sample_maker(test_images, test_labels,sample_size)

# rotated test image build
rotate_degree = 90
np.random.seed(0)
degree = np.random.randint(-rotate_degree,rotate_degree,len(X_test))

for index, img in enumerate(X_test):    
    # path =  str(sample_size) + 'size' + str(rotate_degree) + 'degree/' + str(y_test[index])
    path =  str(sample_size) + 'size/' + str(y_test[index])
    mkdir(path)
    # img = random_rotate(img, degree[index])
    cv2.imwrite(path + '/' + "{:05d}.png".format(index), img)

print('Finished split.')

# sample_size = 20

# X_train, y_train = balanced_sample_maker(train_images, train_labels,sample_size)

# for index, img in enumerate(X_train):    
#     path =  str(sample_size) + 'size/' + str(y_train[index])
#     mkdir(path)
#     misc.imsave(path + '/' + "{:05d}.png".format(index), img)
