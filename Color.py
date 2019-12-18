#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 23:27:57 2019

@author: DavidFelipe
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
#%matplotlib inline


class Color:
    def __init__(self, image):
        self.subset_image = image
        plt.rc('axes', **{'grid': False})
        plt.style.use('ggplot')
        
    def plot_pixels(self, data, title, colors=None, N=10000):
        if(data.max() > 200):
            data = data / 255.0 # use 0...1 scale
            data = data.reshape((-1, 3))
        if colors is None:
            colors = data
        
        # choose a random subset
        rng = np.random.RandomState(0)
        i = rng.permutation(data.shape[0])[:N]
        colors = colors[i]
        pixel = data[i].T
        R, G, B = pixel[0], pixel[1], pixel[2]
        
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))
        ax[0].scatter(R, G, color=colors, marker='.')
        ax[0].set(xlabel='Red', ylabel='Green', xlim=(0, 1), ylim=(0, 1))
    
        ax[1].scatter(R, B, color=colors, marker='.')
        ax[1].set(xlabel='Red', ylabel='Blue', xlim=(0, 1), ylim=(0, 1))
    
        fig.suptitle(title, size=20);
        
    def clustering(self, clusters, verbose=0):
        """
        Input : 
            
            Clusters - Number of clusters to group the colors
        
        Output :
            
            Image_clusterized - Image with binarized colors
        """
        img_data = self.subset_image / 255.0 # use 0...1 scale
        img_data = img_data.reshape((-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        flags = cv2.KMEANS_PP_CENTERS
        compactness, labels, centers = cv2.kmeans(img_data.astype(np.float32),
                                          clusters, None, criteria, 10, flags)
        new_colors = centers[labels].reshape((-1, 3))
        image_recolored = new_colors.reshape(self.subset_image.shape)
        if(verbose==1):
            ## See the cluster process segmentation
            self.plot_pixels(self.subset_image, title='Input color space: 16 million possible colors')
            self.plot_pixels(image_recolored, title='Input color space: 16 million possible colors')
            
        return image_recolored
        
    def sharpening(self, image):
        bil_image = cv2.bilateralFilter(image,10,100,100) 
        self.subset_image = bil_image

        
        
        
        