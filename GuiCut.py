#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 17:14:59 2019
@author: DavidFelipe
"""

import cv2 
import numpy as np
import slidingwindow as sw
import yaml
import time

class GuiCut:
    def __init__(self, image, divide=0, classes=1, max_blocks = 5, path_conf="./"):
        self.image_original = image
        self.max_blocks = max_blocks
        self.raw_image_original = np.copy(image)
        self.activate = False
        self.mode = "default"
        self.drawing = False
        self.config_file(path_conf)
        self.ref_point = []
        self.class_mode = 1
        self.counter_blocks = 0
        self.blocks = []
        if(divide != 0):
            self.windows = sw.generate(self.image_original, sw.DimOrder.HeightWidthChannel, divide, 0.1)
            self.divide = True
#            print(len(self.windows))
        else:
            self.windows = image
            self.divide = False
        self.classes = []
        self.crop_count = np.array([])
        self.classes_key = []
        self.color_sharp_mode = []
        counter = 1
        for i in range (0,classes):
            rectangle = []
            color = np.random.choice(range(256), size=3)
            color_c = (int(color[0]), int(color[1]), int(color[2]))
            self.classes.append(rectangle)
            self.color_sharp_mode.append(color_c)
            self.crop_count = np.append(self.crop_count,0)
            self.classes_key.append(counter)
            counter += 1
            
        
    def mouse_callback(self, event, x, y, flags, param):
        # grab references to the global variables 
        # if the left mouse button was clicked, record the starting 
        # (x, y) coordinates and indicate that cropping is being performed
        if(self.activate):
            if(self.mode == "default"):
                if event == cv2.EVENT_LBUTTONDOWN:
                    self.ref_point = [(x, y)]
                    self.drawing = True
                    
                elif event == cv2.EVENT_MOUSEMOVE: 
                    if self.drawing==True:
                        a = x 
                        b = y 
                        if a != x | b != y:
                            last = [x+(2*self.line_thickness_def), y+(2*self.line_thickness_def)]
                            crop_part = self.image_raw[self.ref_point[0][1]:last[1], self.ref_point[0][0]:last[0]]
                            self.image[self.ref_point[0][1]:last[1], self.ref_point[0][0]:last[0]] = crop_part
                            cv2.rectangle(self.image, self.ref_point[0], (x,y), (255,0,0), self.line_thickness_draw)
              
                # check to see if the left mouse button was released 
                elif event == cv2.EVENT_LBUTTONUP: 
                    self.drawing = False
                    # record the ending (x, y) coordinates and indicate that 
                    # the cropping operation is finished 
                    self.ref_point.append((x, y)) 
                    crop_part = self.image_raw[self.ref_point[0][1]:self.ref_point[1][1], self.ref_point[0][0]:self.ref_point[1][0]]
                    rectangle = self.classes[self.class_mode - 1]
                    rectangle.append((self.ref_point))
                    self.crop_count[self.class_mode - 1] += 1
                    self.counter_blocks += 1
                    # draw a rectangle around the region of interest 
                    cv2.rectangle(self.image, self.ref_point[0], self.ref_point[1], self.color_sharp_mode[(self.class_mode-1)], self.line_thickness_draw)
#                    cv2.imshow("image", self.image) 

    def config_file(self, path):
        with open("config.yml", 'r') as ymlfile:
            config_file = yaml.load(ymlfile, Loader=yaml.FullLoader)
        guicut_conf = config_file['guicut']
        self.line_thickness_def = guicut_conf["line_thickness_def"]
        self.line_thickness_draw = guicut_conf["line_thickness_draw"]
        self.multiplier_erase = guicut_conf["multiplier_erase"]
        self.max_blocks = guicut_conf["max_blocks"]
        
    def classes_handle(self, key):
        for item in self.classes_key:
            if(key == ord(str(item))):
                self.class_mode = item
                print(item)
                break
    
    def generate_sizes(self, class_mode = 0, verbose = 0):
        rectangle = self.classes[class_mode]
        mean_size = np.array([])
        for item in rectangle:
            h = item[1][1] - item[0][1]
            w = item[1][0] - item[0][0]
            prom = (h+w)/2
            mean_size = np.append(mean_size, prom)
            self.blocks.append((int(prom),int(prom)))
        return mean_size.mean()
            
    def delete_item(self,verbose = 0):
        crop_c = self.crop_count[self.class_mode - 1]
        if(crop_c>0):
            rectangle = self.classes[self.class_mode - 1]
            pos = len(rectangle) - 1
            if(len(rectangle)>0):
                limits = rectangle[pos]
                crop_sustitute = self.image_raw[limits[0][1]:(limits[1][1]+self.multiplier_erase*self.line_thickness_def), limits[0][0]:(limits[1][0]+self.multiplier_erase*self.line_thickness_def)]    
                self.image[limits[0][1]:(limits[1][1]+self.multiplier_erase*self.line_thickness_def), limits[0][0]:(limits[1][0]+self.multiplier_erase*self.line_thickness_def)] = crop_sustitute
                del rectangle[pos]
                crop_c -= 1
                self.counter_blocks -= 1
                
    def crop_part_image_random(self):
        while True:
            random_num = (np.random.choice(range(len(self.windows)), size=1))[0]
            part_window = self.windows[random_num]
            part_image = self.image_original[part_window.indices()]
            if(part_image.mean() < 225):
                break
        return part_image
    
    def crop_part_image(self, number):
        while True:
            part_window = self.windows[number]
            part_image = self.image_original[part_window.indices()]
            if(part_image.mean() < 225):
                break
        return part_image
        
    def run(self, verbose=0):
        cv2.namedWindow("image",cv2.WINDOW_FULLSCREEN) 
        cv2.setMouseCallback("image", self.mouse_callback)
        
        self.counter_windows = 0
        flag = False
        for image_subset in self.windows:
            if(self.divide):
                self.image = self.image_original[image_subset.indices()]
            else:
                self.image = self.image_original
            self.image_raw = self.image.copy()
            
            if(self.image.mean() > 225):
                continue
            self.counter_windows += 1
            # keep looping until the 'q' key is pressed 
            while True: 
                
                # display the image and wait for a keypress 
                cv2.imshow("image", self.image) 
                key = cv2.waitKey(1) & 0xFF
              
                # press 'r' to reset the window 
                self.classes_handle(key)
                """
                if(self.counter_blocks == self.max_blocks):
                    self.image = self.image_raw.copy()
                    cv2.imshow("image", self.image)
                    time.sleep(2)
                    flag = True
                    break
                    """
                
                if key == ord("o"):
                    self.image = self.image_raw.copy()
                    cv2.imshow("image", self.image)
                    time.sleep(2)
                    flag = True
                    break
                
                if key == ord("r"): 
                    self.image = self.image_raw.copy()
                    self.ref_point = []
                
                elif key == ord("a"): 
                    print(self.activate)
                    if(self.activate):
                        self.activate = False
                    else:
                        self.activate = True

                elif key == ord("e"):
                    print("Deleted")
                    self.delete_item()
                    
                elif key == ord("h"):
                    counter = 0
                    for item in self.classes:
                        print("in Class " + str(counter)+ ": " +str(len(item)))
                        counter += 1
                    print("Mode :" + str(self.class_mode))
                    print("Color " + str(self.color_sharp_mode[self.class_mode-1]))
                    print("Window #"+str(self.counter_windows)+" de " + str(len(self.windows)))
                
                    # if the 'c' key is pressed, break from the loop 
                elif key == ord("q"):
                    self.image = self.image_raw.copy()
                    break
            if flag:
                self.image = self.image_raw.copy()
                break
        mean_size = int(self.generate_sizes())
        mean_size = [(mean_size,mean_size)]
        cv2.destroyAllWindows()
        return self.blocks, mean_size
        