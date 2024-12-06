import matplotlib.pyplot as plt
import numpy as np
import collections
import random
import math
import cv2
import time

# Load MNISt data.
def load_data(path="MNIST_data/mnist.npz"):
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)

# Pad img to dedicate size with black side.
def padding(img, new_shape=(32, 32), color=(0, 0, 0)):
    # Resize image to a 32-pixel rectangle 
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    dw, dh = new_shape[1] - shape[1], new_shape[0] - shape[0]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img

# Rotate the digit with degree.
def random_rotate(img, degree):
    rows,cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),degree,1) #第三个参数：变换后的图像大小 
    img = cv2.warpAffine(img,M,(rows,cols))    
    return img

# Digit overlap checking   
def isOverlap(r1,r2):
    if(r1[2]<=r2[0]) or (r1[3]<=r2[1]) or (r1[0]>=r2[2]) or (r1[1]>=r2[3]):
        overlap =  False
    else:
        overlap = True
    return overlap 

def write_txt(label_contents,path):
    real_label = open(path,'w')
    for line in label_contents:
        s = " ".join(map(str,line))
        real_label.write( s +'\n')
    real_label.close()
    return 

def find_bbox(img):
    contours,_ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
 
    # Find the index of the largest contour
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]
    (x, y, w, h) = cv2.boundingRect(cnt)
    labels = [x, y, w, h]
    return labels

def load_mix(img,label,shape = (608,608) ,rotate=False):
    
    global sequence

    labels = list()
    cord = list()
    
    canvas = np.zeros(shape)
    quantity = random.randint(1,9)  # Set quatity of digits   
    
    # Put digit into canvas
    for i in range(quantity):

        overlap = list()
        overlap_list = True
        scale = random.random()*5
        if scale >1: # Check scale <1 or not.
            pass 
        else:
            scale = 1  # set enlarge size of digit

        if not sequence:
            print("empty")
            sequence = list(range(len(img)))

        indices = random.choice(sequence)  # Choose image in MINST by index
        sequence.remove(indices)        
        
        img_shape = img[indices].shape  # Pick up digit
        img1 = cv2.resize(img[indices], (int(img_shape[1]* scale), int(img_shape[0]* scale)), interpolation=cv2.INTER_CUBIC)  # Resize the digit base on random scale
        
        # Rotate the digit if rotate is True.
        if rotate:
            degree = random.randint(-45,45)
            img1 = random_rotate(img1, degree)

        bbox  = find_bbox(img1)  # Output x, y w, h by digit cordinate
        size = img1.shape
        xx = math.ceil(random.random()*(shape[1]-size[1]))  # Define x coordinate of digit
        yy = math.ceil(random.random()*(shape[0]-size[0]))  # Define y coordinate of digit        
        
        # Determine if r1, r2 overlap. 
        if i >= 1:            
            while overlap_list:                
                for j, xxyy in enumerate(cord):                                                
                    rec_overlap = isOverlap(xxyy,[xx,yy,xx+size[1],yy+size[0]])
                    if rec_overlap: # If overlap, create new coordinate.
                        xx = math.ceil(random.random()*(shape[1]-size[1]))  # Define x coordinate of digit
                        yy = math.ceil(random.random()*(shape[0]-size[0]))  # Define y coordinate of digit
                        overlap = list()
                        break
                    else:
                        overlap.append(0)
                        if len(overlap) == len(cord):
                            overlap_list = all(overlap)
                                       
        # Print(i,len(cord),len(overlap),overlap)  # Check the overlap lense math with digit quanties.
        cord.append([xx,yy,xx+size[1],yy+size[0]]) 
        canvas[yy:yy+size[0],xx:xx+size[1]] = img1

        # Tranfer to yolo format (center_x,center_y, w, h of normalized values)
        c_x, c_y = ((xx + bbox[0]) + bbox[2]/2)/shape[0] ,((yy + bbox[1]) + bbox[3]/2)/shape[1]
        ww , hh = bbox[2]/shape[0]   , bbox[3]/shape[1]
               
        labels.append([label[indices], c_x, c_y, ww, hh])  # Append label in one list to write in txt       

    return canvas,labels

def plot_img(X):
    plt.imshow(X, cmap='gray')
    plt.show()


start = time.time()

# Train data path
train_img_path = "image/train/"
train_label_path = "label/train/"

# Test data path
test_img_path = "image/test/"
test_label_path = "label/test/"

# Rotate data path
rotate_img_path = "image/rotate/"
rotate_label_path = "label/rotate/" 

(x_train, y_train), (x_test, y_test) = load_data()

# Create train images
# sequence = list(range(len(x_train)))
# for i in range(12100):
#     img, labels = load_mix(x_train,y_train)
#     cv2.imwrite(train_img_path + "{:05d}.jpg".format(i),img)
#     write_txt(labels,train_label_path + "{:05d}.txt".format(i))
#     print(i+1,'/12100', ' Remaining digit:', len(sequence))


# Create test images
# sequence = list(range(len(x_test)))
# for i in range(2000):
#     img, labels = load_mix(x_test,y_test)
#     cv2.imwrite(test_img_path + "{:05d}.jpg".format(i),img)
#     write_txt(labels,test_label_path + "{:05d}.txt".format(i))
#     print(i+1,'/2000', ' Remaining digit:', len(sequence))

# Create rotate images
sequence = list(range(len(x_test)))
for i in range(2000):
    img, labels = load_mix(x_test,y_test,rotate=True)
    cv2.imwrite(rotate_img_path + "{:05d}.jpg".format(i),img)
    write_txt(labels,rotate_label_path + "{:05d}.txt".format(i))
    print(i+1,'/2000', ' Remaining digit:', len(sequence))

end = time.time()
print('Cost: ', end-start,'s')

# a = collections.Counter(y_train)
# b = collections.Counter(y_test)

