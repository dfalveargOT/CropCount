#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 13:09:37 2019

@author: DavidFelipe
"""

import cv2 
import slidingwindow as sw
import numpy as np

class SoftCut:
    
    def __init__(self, image, verbose=0):
        self.image_class = image
        self.shape = image.shape
        self.ref_points = []
        self.verbose = verbose
        
    def get_grid(self, distance_points=5, overlap=0, flag_output = False, color = (255,0,0)):
        """
        Input:
            distance_points - Distance between two consecutive points
            overlap - For window generation recomended 0
            flag_output - Draw the points in a image copy
            color - color to draw points
        
        return : 
            points - List of tupla with the (x,y) point position
            draw_image - Copy image with the points drawed
            
        """
        windows = sw.generate(self.image_class, sw.DimOrder.HeightWidthChannel, distance_points, overlap)
        self.ref_points = []
        subset = np.copy(self.image_class)
        for window in windows:
            indices = window.indices()
            cx = indices[0].start
            cy = indices[1].start
            self.ref_points.append((cx,cy))
            if(flag_output):
                cv2.circle(subset, (cx,cy), 2, (255,0,0), 1)
        
        return self.ref_points, subset
    def subset_extract(self, point, size):
        """
        Input:
            point - tupla of point location 
            size - tupla height and width of the image to extract
        
        Output:
            subset - Image subset with a corner in the point given and sizes given
        """
        size1 = size[0]
        size2 = size[1]
        limit = [point[0]+size1, point[1]+size2]
        if(limit[0] > self.shape[0] or limit[1] >  self.shape[1]):
#            print("limit",limit)
#            print(self.shape)
            if(self.verbose == 1):
                print("Point no valid for dimension exceed")
                print("Sizes{1,2} exceed the image dimesion")
            subset = -1
        else:
            subset = self.image_class[point[0]:limit[0], point[1]:limit[1]]
        return subset
    
    def windows_extract(self, point, sizes):
        """
        Input:
            point - tupla of point location 
            sizes - list tuplas with height and width of the image to extract (x,y)
        Output:
            subset - Image subset with a corner in the point given and sizes given
        """
        subset_images = []
        for item_size in sizes:
            subset_size = self.subset_extract(point,item_size)
            if type(subset_size) != int:
                subset_images.append(subset_size)
            
        return subset_images
    




