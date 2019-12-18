#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 08:12:19 2019
Geotiff process module 
GDAL based
@author: datarock
"""

from osgeo import gdal
import os
import numpy as np
import progressbar

class GeoProcess:
    def __init__(self):
        self.transform = []
        self.image = []
        self.coord_array = []
        
    def load_tiff(self, path_images):
        """
        load_tiff:
            Function to load geotiff files
        
        Input:
            path_images - location of the tiff file
            
        Output:
            image - numpy array image
            name  - name of the image file
        
        """
        files = os.listdir(path_images[0])
        flag = False
        print(files)
        for item in files:
            tiff_flag = item.find(".tif")
            ovr_flag = item.find(".tif.ovr")
            xml_flag = item.find(".tif.aux.xml")
            if(ovr_flag == -1 and xml_flag == -1 and tiff_flag != -1):
                image_name = item
                flag = True
                break
        if flag:
            dataset = gdal.Open(path_images[0]+image_name)
            cols = dataset.RasterXSize
            rows = dataset.RasterYSize
            self.transform = dataset.GetGeoTransform()
            data = dataset.ReadAsArray(0, 0, cols, rows)
            image = data.transpose(1,2,0)
            shape = image.shape
            if shape[2] > 3:
                self.image = np.delete(image,[3],axis=2)
            else:
                self.image = image
            name = "DC_" + image_name[:tiff_flag]
            return self.image.copy(), name
        else:
            return -1, -1
        
    def pixel_conversion(self, vector_points):
        """
        pixel_conversion:
            Function to convert pixel tuple coordinates to
            georeferenced coordinates
        
        Input:
            vector_points - vector of pixel coordinates (px,py)
        
        Output:
            coord_array - Array of coordinates georeferenced
            
        """
        widgets = [progressbar.Percentage(),
            ' ', progressbar.Bar(),
            ' ', progressbar.ETA()]
            
        xOrigin = self.transform[0]
        yOrigin = self.transform[3]
        pixelWidth = self.transform[1]
        pixelHeight = -self.transform[5]
        self.coord_array = np.array([0,0,0])
        print("   &&& GeoProcess generate coordinates ")
        bar = progressbar.ProgressBar(widgets=widgets, maxval=len(vector_points)-1)
        bar.start()
        for item, point in enumerate(vector_points):
            coord_x = xOrigin + point[0]*pixelWidth
            coord_y = yOrigin - point[1]*pixelHeight
            coord_point = [item, coord_x, coord_y]
            self.coord_array = np.vstack((self.coord_array, coord_point))
            bar.update(item)
        bar.update(len(vector_points)-1)
        self.coord_array = np.delete(self.coord_array, [0], axis=0)
        
        return self.coord_array

