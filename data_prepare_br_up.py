#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created:    Tue Apr 25 23:52:27 2023
Updated:    11/14/2023

@author: zahid + matthew
"""

#%% libraries  zahid
import os

import matplotlib.pyplot as plt

import numpy as np

import cv2
import cv2 as cv

import glob

from scipy.io import loadmat
from scipy.signal import butter, filtfilt, buttord
from scipy.fft import fft, fftfreq

import random

from random import seed, randint

from sys import exit

# from sklearn.model_selection import train_test_split

import pandas as pd
import pickle
import pdb

#%%  Data Load files from the directory Zahid

# Select the source file [either MERL or Collected Dataset or ]


# load Pathdir
#iD_ir = '../../../Dataset/Merl_Tim/Subject1_still/IR'
#iD_ir = '../../../Dataset/Merl_Tim/Subject1_still/RGB_raw'
#iD_ir = '../../../Dataset/Merl_Tim/Subject1_still/RGB_demosaiced'

path_dir = 'data_/'

dataPath = os.path.join(path_dir, '*.mp4')


files = glob.glob(dataPath)  # care about the serialization
# end load pathdir
list.sort(files) # serialing the data

if not files:
    raise Exception("Data upload failure!")

# Take time stamp and multiple by 64. Take starting time of the BVP file, 
# subtract the tags.csv from the BVP start time, multiply by 64 to get the sample number. 


#%% Load the Video and corresponding zahid

# find start position by pressing the key position in empatica
# perfect alignment! checked by (time_eM_last-time_eM_first)*30+start_press_sample  should
# give the end press in video


def test_video(files, im_size=(200, 200)):
    # begin video capture
    f_num = 22
    print(files[f_num])
    cap = cv2.VideoCapture(files[f_num])
        
    ijk = 0

    while(cap.isOpened()):
        ret, frame1 = cap.read()
        
        if ret==False:
            break
            
        # pdb.set_trace()
        if (ijk == 2000):
              breakpoint()  
        ijk += 1
        
        frame1 = frame1[:,:,:]
        frame1 = cv2.resize(frame1, im_size)

        img_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (5,5), 0) 
         
        # Canny Edge Detection
        edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=100) # Canny Edge Detection
        
        # Display Canny Edge Detection Image
        cv2.imshow('Canny Edge Detection', edges)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    
def data_read_edge_detection(files, im_size=[300, 300]):
    data = []
    data_end = len(files)
    
    # print files list
    print(*(x for x in files), sep='\n')
    print('\n')
    
    # take file input and handle issues
    try:
        f_num = int(input("Video Sample Request: "))
        if (f_num < 0 or f_num > data_end):
            raise Exception("Bad Video Request ...")
            
    except:
        raise Exception("Bad Video Request ...")
    
    # bank of manual annotations for video sample ROIs 
    # (sample no. : [[rows], [cols]])
    """BAD BANK :(      [21]"""
    annotation_bank = {1: [[500, 750], [250, 750]], 2: [[500, 750], [250, 750]], 
                       0: [[500, 950], [200, 750]], 4: [[500, 750], [350, 800]],
                       2: [[1000, 1600], [300, 1600]], 6: [[0, 150], [500, 550]],
                       3: [[100, 1200], [100, 1500]], 8: [[850, 1200], [250, 900]],
                       9: [[1100, 1400], [0, 600]], 10: [[450, 650], [0, 350]],
                       11: [[800, 1000], [350, 1000]], 12: [[1200, 1450], [0, 600]],
                       13: [[400, 550], [100, 300]], 14: [[1000, 1300], [400, 1000]],
                       15: [[275, 400], [150, 400]], 16: [[300, 450], [50, 250]],
                       17: [[550, 650], [100, 350]], 18: [[1100, 1400], [150, 900]],
                       19: [[1200, 1400], [300, 800]], 20: [[1000, 1300], [0, 200]],
                       21: [[0, 0], [0, 0]], 22: [[1100, 1250], [500, 650]], 
                       23: [[1100, 1300], [1500, 1900]], 24: [[650, 750], [1400, 1600]],
                       25: [[750, 900], [900, 1000]], 26: [[600, 850], [1200, 1350]],
                       27: [[500, 650], [600, 850]], 28: [[550, 700], [450, 600]]}
    
    # begin video capture
    cap = cv2.VideoCapture(files[f_num])
    frame_num = 0
    
    valss= np.zeros((200,200,3))
    
    # import pdb
    ijk = 0
    if (ijk == 0):
        print(files[f_num])
        
    

    while(cap.isOpened()):
        ret, frame1 = cap.read()
        
        if ret==False:
            break
            
        # pdb.set_trace()
        # if (ijk == 1000):
        #       breakpoint()
            
        # ijk += 1
        
        # take manual annotations for selected video data
        row_1 = annotation_bank[f_num][0][0]
        row_2 = annotation_bank[f_num][0][1]
        col_1 = annotation_bank[f_num][1][0]
        col_2 = annotation_bank[f_num][1][1]
        
        frame1 = frame1[:,:,:]
        frame1 = frame1[row_1:row_2, col_1:col_2] 
        # im_size[1] = row_2 - row_1
        # im_size[0] = col_2 - col_1
        # im_size = tuple(im_size)
        
        frame1 = cv2.resize(frame1, im_size)
        im_size = list(im_size)

        img_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        
        # img_blur = cv2.GaussianBlur(img_gray, (5,5), 0) 
        
        kernel = np.ones((3,3),np.float32)/9  # parameter change 1  # play with it
        img_blur = cv.filter2D(img_gray,-1,kernel)
         
        # Canny Edge Detection
        
        # play with threshold values
        
        edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=100) # Canny Edge Detection
        # Display Canny Edge Detection Image
        data.append(frame1)
        
        kernel = np.ones((3,3),np.uint8)
        kernel1 = np.ones((5,5),np.uint8)
        
        if (frame_num >= 1500 & frame_num%5 == 0):
            # edges = cv2.dilate(edges,kernel,iterations = 1) # play with it remove or keep it
            # edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel1)
            cv2.imshow('Canny Edge Detection', edges)
            # pdb.set_trace()
        # ijk += 1
        # pdb.set_trace()
        
        frame_num += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    
    # fps = cap.get(cv2.CAP_PROP_FPS)
        
    cap.release()
    cv2.destroyAllWindows()
    data =  np.array(data)
    
    return data

# test_video(files)
data_og = data_read_edge_detection(files, im_size=[640, 360])
data = data_og

plt.imshow(data[100])

data = data[400:,:,:] # select a good starting frames 400, 800, check by plt.imshow

#%% Filtering functions matthew






