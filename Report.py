#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 22:30:08 2019
Copyright © 2019 DataRock S.A.S. All rights reserved.
@author: DavidFelipe
"""


try:
    import time
    import yaml
    import cv2
    import os
    import numpy as np
except:
    print(" PLEASE REVIEW THE MODULES THAT NEEDS THE SOFTWARE - AN ERROR WAS OCCURRED REPORT MODULE")

#
#print(" %% REPORT MODULE %%")
#print(" -- Report of counting process  -- Check the current progress --")

class Report:
    def __init__(self):
        self.images = []
        self.path_images = ""
        self.path_results = ""
        self.show_software_info()
    
    def load_image(self, path_images):
        files = os.listdir(path_images[0])
        flag = False
        for item in files:
            jpg_flag = item.find(".jpg")
            png_flag = item.find(".png")
            jpeg_flag = item.find(".jpeg")
            if(jpg_flag != -1 or png_flag != -1 or jpeg_flag != -1):
                image_name = item
                flag = True
                break
        if flag:
            self.image = cv2.imread(path_images[0]+image_name)
            name = "DC_"+image_name
            return self.image, name
        else:
            return -1, -1
    
    def config_file(self, path="./"):
        with open("config.yml", 'r') as ymlfile:
            config_file = yaml.load(ymlfile, Loader=yaml.FullLoader)
        main_conf = config_file['main']
        self.path_images = main_conf["path_images"]
        self.path_results = main_conf["path_results"]
        paths = [self.path_images, self.path_results]
        return paths
    
    def show_software_info(self):
        
        print('\n')
        print("DATAROCK S.A.S")
        print("CROPS COUNTING SOFTWARE")
        print("Copyright © 2019 DataRock S.A.S. All rights reserved.")
        print("Developed by David Felipe Alvear Goyes")
        print('\n')
    
    def Generate_csv(self, matrix,  name):
        np.savetxt(self.path_results + name + "_bx.csv", matrix, delimiter=";",fmt=['%d' , '%f', '%f'])

    def Generate(self, name, path, image_time, count):
        """
        Generate the current image report of the counting
        """
        results = open(path + "Results_"+ name +".txt", "w+")
        strE = "DATAROCK S.A.S\n"
        strC = "Copyright © 2019 DataRock S.A.S. All rights reserved.\n"
        strF = time.ctime() + "\n"
        str1 = "Results for " + name + "\n"
        str2 = "Total count : " + str(count) + "\n"
        str3 = "Total time of image process : " + str(round(image_time,1)) + "Seconds" + "\n"
        strN = "Software by David Alvear G. \n"
        results.write(strE)
        results.write(strC) 
        results.write(strF)    
        results.write(str1)
        results.write(str2)
        results.write(str3)
        results.write(strN)
        results.close()
        
        