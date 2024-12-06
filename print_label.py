import os
import sys
import cv2
# import numpy as np

def change_cor(position,shape):
    position = list(map(float,position))
    x1 = int((position[0] - (1/2)*position[2])*shape[1])
    y1 = int((position[1] - (1/2)*position[3])*shape[0])
    x2 = int((position[0] + (1/2)*position[2])*shape[1])
    y2 = int((position[1] + (1/2)*position[3])*shape[0])

    coordinate = x1,y1,x2,y2
    return coordinate 


# 獲取當前資料夾的路徑
path = os.getcwd()

# 要修改的檔案的格式
pic_ext = ['.jpg','.png','.PNG','.jpeg','.gif','.JPG','.JPEG']
num = 0

image_name = os.listdir('image/rotate')
label_name = os.listdir('label/rotate')

# print(image_name)
# print(label_name)

image_name.sort()
image_name.sort()

# f = open('list.txt','w')
# for img in image_name:
#     f.write(path+'\\train_images'+ '\\' + img + '\n')
# f.close()

for index, img_name in enumerate(image_name):
    
    image = cv2.imread('image/rotate/' + img_name)

    img_name,img_ext = os.path.splitext(img_name)
    label_path = 'label/rotate/'+ img_name + '.txt'
    print(img_name)
    # cv2.imshow('image',image)
    # cv2.waitKey()
    shape = image.shape

    lines = []
    with open(label_path) as f:
        lines = f.readlines()
        
    for data in lines:
        data = data.split()
        class_str = data[0]
        data.pop(0)
        cordinate = change_cor(data,shape)
        image = cv2.rectangle(image,(cordinate[0],cordinate[1]),(cordinate[2],cordinate[3]),(255,0,0),thickness=2)

    cv2.imwrite('label_img_test\\' + str(img_name) + ".jpg",image)


# for file in os.listdir(path):

#     if os.path.isfile(file) == False:
#         for filename in os.listdir(path + "/" + file):
#             if os.path.isfile(path + "/" + file + "/" + filename) == True:
#                 name,ext = os.path.splitext(filename)
#                 if ext in pic_ext:
#                     newname = str(num) + ".jpg"
#                     os.rename(path + "/" + file + "/" + filename , path + "/" + file + "/" + newname)
#                     num += 1
#             else:
#                 for filename_1 in os.listdir(path + "/" + file + "/" + filename):
#                     if os.path.isfile(path + "/" + file + "/" + filename + "/" + filename_1) == True:
#                         name,ext = os.path.splitext(filename_1)
#                         if ext in pic_ext:
#                             newname = str(num) + ".jpg"
#                             os.rename(path + "/" + file + "/" + filename + "/" + filename_1 , path + "/" + file + "/" + filename + "/" + newname)
#                             num += 1

# print(num)