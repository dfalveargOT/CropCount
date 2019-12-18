#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 11:20:16 2019

@author: DavidFelipe
"""
#try:
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import progressbar
import SubsetCut
import SoftCut
import GuiCut
import DeepClasifier
from Postprocessing import Postprocessing
from Report import Report
from GeoProcess import GeoProcess
import time
#except:
#    print(" PLEASE REVIEW THE MODULES THAT NEEDS THE SOFTWARE - AN ERROR WAS OCCURRED MAIN")
#
#    
widgets = [progressbar.Percentage(),
            ' ', progressbar.Bar(),
            ' ', progressbar.ETA()]

"""
initialization
"""
manager = Report()
paths = manager.config_file()
geo_manager = GeoProcess()
image, name = geo_manager.load_tiff(paths)
if type(name) == int:	
    print("Not found valid format to start the process ... ")
    time.sleep(3)
    exit()

"""
cut procedure
"""
print(" &&& Subsetcut in")
subsetcut = SubsetCut.subsetcut(image, fit=True)
subsetcut.run()
image_segmented = subsetcut.masked_image
print(" &&& Subsetcut out")
print(" ")
"""
Tool definitions
"""
print(" &&& Guicut in")
guicut = GuiCut.GuiCut(image_segmented, divide=1000, classes=2)
print(" &&& Subsetcut out")
print(" ")
print(" &&& softcut in")
softcut = SoftCut.SoftCut(image_segmented, verbose=0)
print(" &&& Softcut out")
print(" ")
print(" &&& DeepClasifier in")
deepclasifier = DeepClasifier.DeepClasifier()
deepclasifier.load_model()#    exit()
print(" &&& DeepClasifier out")
print(" ")
"""
Configurations
"""
print(" &&& guicut run")
anchor_boxes, mean_size = guicut.run()
points_grid, _ = softcut.get_grid(distance_points=15, flag_output=False) ## Default 10
print(" &&& guicut off")
print(" ")

"""
Generate the list of total images to evaluate the terrain
"""
image_time = time.time()
list_subsets = []
good_points = []
print(" &&& Making deep vectors")
time_ = time.time()
bar = progressbar.ProgressBar(widgets=widgets, maxval=len(points_grid)-1)
bar.start()
for item, point in enumerate(points_grid):
    subset_images = softcut.windows_extract(point, anchor_boxes)
    if len(subset_images) >= 1 and subset_images[0].any() != 0:
        list_subsets.append(subset_images[0])
        good_points.append(point)
    bar.update(item)
bar.update(len(points_grid)-1)
print("  &&& Time making deep vectors elapsed : " + str(time.time() - time_) + " Seconds")
print(" ")

"""
Predict all the batch of images throught gpu processing
"""
time_ = time.time()
print(" &&& Create list")
list_batches = deepclasifier.prepare_list(list_subsets, size = 600)
print(" &&& Predict list")
result = deepclasifier.predict_batches(list_batches)
print("  &&& Time DL elapsed : " + str(time.time() - time_) + " Seconds")
print(" ")
"""
Create the array of boxes 
"""
print(" &&& Clean results vector")
time_ = time.time()
bar = progressbar.ProgressBar(widgets=widgets, maxval=len(good_points)-1)
bar.start()
for item, point in enumerate(good_points):
    score = result[item]
    deepclasifier.check_detector(point, anchor_boxes[0], score)
    bar.update(item)
bar.update(len(good_points)-1)
print("  &&& Time Clean results elapsed : " + str(time.time() - time_) + " Seconds")
print(" ")

"""
Postprocessing
"""
print(" &&& finalprocess")
time_ = time.time()
finalprocess = Postprocessing(image, deepclasifier.vector_boxes)
print("  &&& Time finalprocess elapsed : " + str(time.time() - time_) + " Seconds")
print(" ")

'''
pointed_image = finalprocess.container[0]
boxed_image = finalprocess.image_drawed[0]
'''

"""
TEMPORAL
"""
image_draw = finalprocess.Draw_results(image, finalprocess.bboxes_after_nms)


"""
Generate Report
"""
print(" &&& Report information")
time_ = time.time()
geo_manager.pixel_conversion(finalprocess.bboxes_after_nms)
print("  &&& Time Generate coord_array elapsed : " + str(time.time() - time_) + " Seconds")
print(" ")
Total_time = time.time() - image_time
manager.Generate(name, paths[1], Total_time, len(finalprocess.bboxes_after_nms))
manager.Generate_csv(geo_manager.coord_array, name)
print(" ")
finish = input(" Push any key to finish ")
#exit()
