#!/usr/bin/env python
# coding: utf-8

# # Install and Import Dependencies
# 

# In[1]:


get_ipython().system('pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118')


# In[2]:


get_ipython().system('git clone https://github.com/ultralytics/yolov5')


# In[8]:


get_ipython().system('cd yolov5 & pip install -r requirements.txt ')


# In[9]:


import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2


# # Load Model

# In[4]:


model=torch.hub.load("ultralytics/yolov5",'yolov5s')


# In[5]:


model


# In[6]:



img = 'https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEiRkw0eO4ykNkJZYm4sga6wl2Vx0CP5j3LmI-b12hCBCoq2vTVy8UnzcAb9sCtVnTjEnOfvPKlOg257OfKsRFHx4rX15DRKawXgkbPmOvZAP8LkmlwGklMbRbWK2A7V6B_KqUSnVlVSnLw/s1600/fruitbowl2b.jpg'


# In[7]:


results=model(img)
results.print()


# In[8]:


#matplotlib inline
plt.imshow(np.squeeze(results.render()))
plt.show()


# In[9]:


results.xyxy


# In[10]:


results.show()


# In[11]:


results.render()


# In[12]:


np.array(results.render()).shape


# In[13]:


np.squeeze(results.render()).shape


# In[14]:


plt.imshow(np.squeeze(results.render()))


# # Real time Detections

# In[15]:


cap= cv2.VideoCapture(0)
while cap.isOpened():
    ret,frame=cap.read()
    
    #Make Detections
    results=model(frame)
    cv2.imshow('YOLO Drowsy Detection',np.squeeze(results.render()))
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# In[17]:


#Detecting video
cap= cv2.VideoCapture('traffic.mp4')
while cap.isOpened():
    ret,frame=cap.read()
    
    #Make Detections
    results=model(frame)
    cv2.imshow('YOLO Drowsy Detection',np.squeeze(results.render()))
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# # Train from Scratch 

# In[16]:


import uuid #unique identifier
import os
import time


# In[17]:


IMAGES_PATH=os.path.join('data','images') #/data/images
labels=["fully_awake","slightly_awake","moderately_awake","mildly_drowsy","partially_drowsy","heavily_drowsy","fully_drowsy"]
number_imgs=5


# In[18]:


cap=cv2.VideoCapture(0)
#Loop through labels
for label in labels:
    print('Collecting images for {}'.format(label))
    time.sleep(5)
    
    #Loop through image range
    for img_num in range(number_imgs):
        print('Collecting images for {}, image number {}'.format(label,img_num))
        
        #Webcam feed
        ret, frame = cap.read()
        
        #Naming out image path
        imgname=os.path.join(IMAGES_PATH, label+'.'+str(uuid.uuid1())+'.jpg')
        
        #Writes out image to file
        cv2.imwrite(imgname,frame)
        
        #Render to the screen
        cv2.imshow('Image Collection',frame)
        
        #2 second delay between captures
        time.sleep(2)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()


# In[1]:


get_ipython().system('git clone https://github.com/HumanSignal/labelImg')


# In[2]:


get_ipython().system('pip install pyqt5 lxml --upgrade')
get_ipython().system('cd labelImg && pyrcc5 -o libs/resources.py resources.qrc')
    


# In[16]:


#training command
get_ipython().system('cd yolov5 && python train.py --img 320 --batch 16 --epochs 100 --data dataset.yml --weights yolov5s.pt')


# # Load Custom Model
# 

# In[24]:


model= torch.hub.load('ultralytics/yolov5','custom',path='yolov5/runs/train/exp3/weights/last.pt')


# In[25]:


cap= cv2.VideoCapture(0)
while cap.isOpened():
    ret,frame=cap.read()
    
    #Make Detections
    results=model(frame)
    cv2.imshow('YOLO Drowsy Detection',np.squeeze(results.render()))
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# In[ ]:




