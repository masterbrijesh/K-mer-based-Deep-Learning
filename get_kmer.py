from typing import Sequence
import numpy as np
import cv2
import math

median_filter = np.ones([3,3]) / 9
gaussian_filter = np.array([[1,2,1],[2,4,2],[1,2,1]]) / 16

median_filter5 = np.ones([5,5]) / 25
gaussian_filter5 = np.array([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]]) / 273


class Kmer_extractor():
    
    def __init__(self, degree=10, k=10, kernel=None, kernel_size = 3, sequence=True):
        # Input shape
        self.img_rows = 28
        self.img_cols = 28
        self.angle_step = degree
        self.knum = k
        self.kernel_size = kernel_size
        self.sequence = sequence

        global median_filter, median_filter5
        global gaussian_filter, gaussian_filter5


        if kernel == 'median':
            self.kernel = median_filter if self.kernel_size==3 else median_filter5
        elif kernel == "gaussian":
            self.kernel = gaussian_filter if self.kernel_size==3 else gaussian_filter5
        else:
            self.kernel = None

        self.new_rows = 0
        self.new_cols = 0
        self.add_x = 0
        self.add_y = 0
        self.pos_array = self.get_kmer_pos()
        

    def createLineIterator(self, start, end, num):
        # num = int(round(((end[1] - start[1])**2+(end[0]-start[0])**2)**0.5))
        x_delta = (end[0] - start[0])/(num-1)
        y_delta = (end[1] - start[1])/(num-1)
        position_list=[]
        if (x_delta >= 1 and y_delta >=1):
            for count in range(num):
                if (count==0):
                    position_list.append(start)
                elif (count==num-1):
                    position_list.append(end)
                else:
                    x_next = int(round(start[0] + x_delta*count))
                    y_next = int(round(start[1] + y_delta*count))
                    pos_next = np.array([x_next,y_next])
                    position_list.append(pos_next)
        else:
                for count in range(num):
                    if (count==0):
                        position_list.append(start)
                    elif (count==num-1):
                        position_list.append(end)
                    else:
                        pos_pre = position_list[count-1]
                        x_f = start[0] + x_delta*count
                        y_f = start[1] + y_delta*count
                        x_int = int(round(x_f))
                        y_int = int(round(y_f))
                        if (x_int==pos_pre[0] and y_int==pos_pre[1]):
                            if (x_delta > y_delta):
                                x_int += 1
                            else:
                                y_int += 1
                        if (x_int==end[0] and y_int==end[1]):
                            if (x_delta > y_delta):
                                y_int-=1
                            else:
                                x_int-=1
                        position_list.append(np.array([x_int,y_int]))
        return position_list
    
    def get_roundpos_list(self):
        img = np.zeros((self.img_rows,self.img_cols))
        img_bit=img.astype(np.uint8).copy()
        MaxR = round(((img.shape[0]/2)**2+(img.shape[1]/2)**2)**0.5)
        if (MaxR % 2 == 0):
            MaxR+=1    
        add_y = round(MaxR-img.shape[0]/2-1) + 1 if self.kernel_size == 3 else round(MaxR-img.shape[0]/2-1) + 2
        add_x = round(MaxR-img.shape[1]/2-1) + 1 if self.kernel_size == 3 else round(MaxR-img.shape[1]/2-1) + 2
        self.add_y = add_y
        self.add_x = add_x
        if (img.shape[0]%2==0):
            img_bit = cv2.copyMakeBorder(img_bit,1,0,0,0,0)
        if (img.shape[1]%2==0):
            img_bit = cv2.copyMakeBorder(img_bit,0,0,1,0,0)
        img_bit = cv2.copyMakeBorder(img_bit,add_y,add_y,add_x,add_x,0)
        center_point = np.array([int((img_bit.shape[1])/2),int((img_bit.shape[0])/2)])
        kmer_pos_all=[]
        for angle in range(0,360,self.angle_step):
            x_end = round((MaxR-1) * math.cos(angle*math.pi / 180) + center_point[0])
            y_end = round((MaxR-1) * math.sin(angle*math.pi / 180) + center_point[1])
            end_point = np.array([int(x_end),int(y_end)])
            sample_points = self.createLineIterator(center_point,end_point,MaxR)
            kmer_pos=[]
            for point in sample_points:
                # kmer_code.append(img_bit[point[1]][point[0]])
                kmer_pos.append(point)
            kmer_pos_all.append(kmer_pos)
        self.new_rows = img_bit.shape[0]
        self.new_cols = img_bit.shape[1]
        return kmer_pos_all
    
    def get_kmerpos_list(self,all_pts):
        kmer_pts = []
        if self.knum <= len(all_pts[0]):
            k_step = (len(all_pts[0])-1)/(self.knum-1)
            for pos_list in all_pts:
                kcode=[]
                for i in range(self.knum):
                    kcode.append(pos_list[int(k_step*i)])
                kmer_pts.append(kcode)
        else:
            for pos_list in all_pts:
                kcode=[]
                for i in range(self.knum):
                    pos = i%len(pos_list)
                    kcode.append(pos_list[pos])
                kmer_pts.append(kcode)
        return kmer_pts
    
    def get_kmer_pos(self):
        all_kmer = self.get_roundpos_list()
        kmer_code = self.get_kmerpos_list(all_kmer)
        kmer_array = np.array(kmer_code)     
        return kmer_array

    def select_sort(self, items, sequence= lambda x,y: x<y):
        sequence_items = items[:].copy()
        items_sum = np.sum(items, axis = 1)
        for i in range(len(items_sum)-1):
            min_index = i
            for j in range(i+1, len(items_sum)):
                if sequence(items_sum[j],items_sum[min_index]):
                    min_index = j
            sequence_items[[i, min_index]] = sequence_items[[min_index, i]]
            items_sum[i], items_sum[min_index] = items_sum[min_index], items_sum[i]     
        return sequence_items

    def kernel_cal(self, img, x, y):                                   
        if self.kernel_size == 3:
            image_region = img[y-1:y+2,x-1:x+2]
        elif self.kernel_size == 5:
            image_region = img[y-2:y+3,x-2:x+3]
        val = np.sum(np.multiply(image_region,self.kernel))

        return val

    def get_kmer_features(self,img):
        img_kmer = img.copy()
        if (img.shape[0]%2==0):
            img_kmer = cv2.copyMakeBorder(img_kmer,1,0,0,0,0)
        if (img.shape[1]%2==0):
            img_kmer = cv2.copyMakeBorder(img_kmer,0,0,1,0,0)
        img_kmer = cv2.copyMakeBorder(img_kmer,self.add_y,self.add_y,self.add_x,self.add_x,0)
        
        kmer_array = list()        
        if len(img_kmer.shape)==2:
            for angles in self.pos_array:
                kmer_list = list()
                for pt in angles:
                    if  self.kernel is None:
                        kmer_list.append(img_kmer[pt[1]][pt[0]])                        
                    else:
                        kmer_list.append(round(self.kernel_cal(img_kmer, pt[0], pt[1]),0))
                kmer_array.append(kmer_list)
            if self.sequence == True:
                kmer_code = self.select_sort(np.array(kmer_array)).reshape(-1) 
            elif self.sequence == False:                
                kmer_code = np.array(kmer_array).reshape(-1)
            elif self.sequence == 'mix':
                sequence_code = np.array(self.select_sort(np.array(kmer_array)).reshape(-1), dtype = np.float64)
                unsequence_code = np.array(np.array(kmer_array).reshape(-1), dtype = np.float64)
                kmer_code = (np.multiply(sequence_code,5)+ np.multiply(unsequence_code,1))/6
            return kmer_code

        elif len(img_kmer.shape)==3:   # 3通道 
            for ch in range(3):
                for angles in self.pos_array:
                    kmer_list = list()
                    for pt in angles:                        
                        kmer_list.append(img_kmer[pt[ch][1]][pt[0]])
                    kmer_array.append(kmer_list)
            return self.select_sort(np.array(kmer_array)).reshape(-1)
        else:
            return -1



