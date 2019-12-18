#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 15:54:43 2019

@author: datarock
"""

import cv2
import os
import DeepClasifier

path = "./x_p/"
files = os.listdir(path)
images = []
for im in files:
    img = cv2.imread(path+im)
    images.append(img)
    
deepclasifier = DeepClasifier.DeepClasifier()
deepclasifier.load_model()#    exit()
result = deepclasifier.predict_batch(images)

