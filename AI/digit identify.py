#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# In[96]:


img = cv.imread(r'C:\Users\user\Desktop\number.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
gray = 255-gray
cv.imshow("binary2",gray) 
cv.waitKey(0)


# In[97]:


# 針對區塊做檢測
kernel = np.ones((4,4),np.uint8)

# 削減
erosion = cv.erode(gray ,kernel,iterations = 1)
# cv.imshow('',erosion)
#  0為igmaX : igmaX小，表现在高斯曲线上就是曲线越高越尖
blurred  = cv.GaussianBlur(erosion,(5,5),0)

# cv.imshow('',blurred  )

# 邊緣檢測 edges = cv2.Canny(gray, low_threshold, high_threshold)
edged= cv.Canny(blurred, 30,150)

# cv.imshow('',edged)
# 擴張
dilation = cv.dilate(edged, kernel ,iterations = 1)
cv.imshow('',dilation)

cv.waitKey(0)


# In[ ]:





# In[100]:


contours, hierarchy = cv.findContours(dilation.copy(),cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
cnts = sorted([(c,cv.boundingRect(c)[0])for c in contours], key = lambda x : x[1])
# cv.imshow("binary2", dilation.copy()) 
# cv.waitKey(0)
#找尋有效值
ary=[]
for (c, _) in cnts:
    (x,y,w,h) = cv.boundingRect(c)
    print(x,y,w,h)
    if w>30 and h>80:
        ary.append(((x,y,w,h)))
        print(ary)


# In[101]:


fig = plt.figure()
# 组合为一个索引序列
num = []
for id, (x,y,w,h) in enumerate(ary):
    roi =erosion[y-20:y+h+20,x-20:x+w+20]
    thresh = roi.copy()
    a = fig.add_subplot(1,len(ary),id+1)
    res = cv.resize(thresh,(28,28))
    num.append(res.tolist())
    cv.imwrite(r'D:\HTML\%d.png'%(id),res)
    plt.imshow(res)
    cv.imshow("binary2", res) 
    cv.waitKey(0)


# In[102]:


nums = np.array(num)

nums = nums.reshape(-1,1,28,28)/255
text_x = torch.tensor(nums, dtype=torch.float32)

# cv.imshow('', num[1])
# cv.waitKey(0)


# In[39]:


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(.3)
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2), 
            nn.Dropout(.3)
        )
      
        self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           
        output = self.out(x)
        return output, x    


# In[40]:


import pandas as pd
net2 = torch.load(r'C:\Users\user\Desktop\save\save11.pt')
net2



# In[103]:



test_output, last_layer = net2(text_x)
pred_y = torch.max(test_output, 1)[1].data.squeeze()
pred_y


# In[ ]:




