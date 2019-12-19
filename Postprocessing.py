#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 10:09:35 2019

@author: DavidFelipe

"""
try:
    import os
    import numpy as np
    import cv2
    import yaml
    import progressbar
    import datetime
except:
    print(" PLEASE REVIEW THE MODULES THAT NEEDS THE SOFTWARE - AN ERROR WAS OCCURRED")

class Postprocessing:
    def __init__(self,image,vector_boxes, draw_flag=False):
        """
        Postprocessing:
            Collection of tools to perform the last modifications to vision 
            computer neural network applications 
        
        Input:
            image - image processed by the neural network
            vector_boxes - vector of coordinates of the object detected boxes
        """
        self.widgets = [progressbar.Percentage(),
            ' ', progressbar.Bar(),
            ' ', progressbar.ETA()]

        boxes_format = np.array([0,0,0,0,0])
        self.vector_points = np.array([0,0])
        vector_boxes = np.delete(vector_boxes, [0], axis=0)
        self.config_file()
        point_image = image
        box_image = image
        if len(vector_boxes > 2):
            print("   &&& Postprocessing change results format")
            bar = progressbar.ProgressBar(widgets=self.widgets, maxval=len(vector_boxes)-1)
            bar.start()
            ## Pass (x,y,w,h) to (x1,y1,x2,y2)
            for item, box in enumerate(vector_boxes):
                xmin = box[1]
                ymin = box[0]
                xmax = box[1] + box[3]
                ymax = box[0] + box[2]
                box_f = np.array([xmin,ymin,xmax,ymax,box[4]])
                boxes_format = np.vstack((boxes_format, box_f))
                bar.update(item)
            bar.update(len(vector_boxes)-1)
            score = boxes_format[:,4]
            boxes = (boxes_format[:,:4]).astype(np.int)
            self.bboxes_after_nms, self.scores_nms = self.NMS_process(boxes,score, self.iou_threshold)
            #self.image_drawed = self.Draw_results(box_image, self.bboxes_after_nms)
            #self.Count_points(point_image, self.bboxes_after_nms)
        else:
            self.image_drawed = point_image
            self.counter = 0
            self.container = [image, image]
        
    def Draw_results(self, image, boxes, mask=0):
        """
        Draw_results:
            Function to draw the boxes provided by the neural network
            Also for create a mask with the center points of the boxes
        
        Input:
            image - Original image processed 
            boxes - vector of coordinates of the boxes found ((Xtopl, Yttopl),(Xbottomr, Ybottomr))
            mask  - (0,1) determine the return object
        
        Output:
            if mask = 0 - Return Image drawed with boxes
            if mask = 1 - Return Mask image with white center points 
        """
        if mask==0:
            for bbox in boxes:
                top_left = bbox[0],bbox[1]
                bottom_right = bbox[2],bbox[3]
                cv2.rectangle(image,top_left, bottom_right,(255, 0, 0), 2)
            return [image]
        else:
            mask_image = np.zeros_like(image, dtype=np.uint8)
            for bbox in boxes:
                cx = int(bbox[0] + (bbox[2] - bbox[0])/2)
                cy = int(bbox[1] + (bbox[3] - bbox[1])/2)
                point = np.array([cx,cy])
                self.vector_points = np.vstack((self.vector_points, point))
                bottom_right = bbox[2],bbox[3]
                cv2.circle(mask_image, (cx, cy), self.radio_mask, (255, 255, 255), -1) #6
                cv2.circle(image, (cx, cy), self.radio_ext, (255, 255, 255), 2) #10
                cv2.circle(image, (cx, cy), self.radio_im, (0, 0, 255), -1) #4
            self.vector_points  = np.delete(self.vector_points , [0], axis=0)
            return [image, mask_image[:,:,0]]
    
    def Count_points(self,image,boxes):
        """
        Count_points:
            Fucntion to draw the centroid of the boxes and count the objects in base of this geometric shape
        Input:
            image - original image
            boxes - vector of boxes already filtered
        Output:
            image_drawed - image with the geometric shapes
            counter - number of objects counted
            
            (container[1]).astype('uint8')
        """
        self.counter = 0
        self.container = self.Draw_results(image, boxes, mask=1)
        mask = self.container[1]
        try:
            _, contours,hierachy = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except:
            contours,hierachy = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for (i, contour) in enumerate(contours):
            self.counter += 1

            
    def NMS_process(self,bboxes,psocres,threshold):
        '''
        NON-MAX-SUPRESSION
        NMS: first sort the bboxes by scores , 
            keep the bbox with highest score as reference,
            iterate through all other bboxes, 
            calculate Intersection Over Union (IOU) between reference bbox and other bbox
            if iou is greater than threshold,then discard the bbox and continue.
            
        Input:
            bboxes(numpy array of tuples) : Bounding Box Proposals in the format (x_min,y_min,x_max,y_max).
            pscores(numpy array of floats) : confidance scores for each bbox in bboxes.
            threshold(float): Overlapping threshold above which proposals will be discarded.
            
        Output:
            filtered_bboxes(numpy array) :selected bboxes for which IOU is less than threshold. 
        '''
        print("   &&& Postprocessing NON-MAX-SUPRESSION")
        #Unstacking Bounding Box Coordinates
        bboxes = bboxes.astype('float')
        x_min = bboxes[:,0]
        y_min = bboxes[:,1]
        x_max = bboxes[:,2]
        y_max = bboxes[:,3]
        
        #Sorting the pscores in descending order and keeping respective indices.
        sorted_idx = psocres.argsort()[::-1]
        #Calculating areas of all bboxes.Adding 1 to the side values to avoid zero area bboxes.
        bbox_areas = (x_max-x_min+1)*(y_max-y_min+1)
        
        #list to keep filtered bboxes.
        filtered = []
        counter = 0
        bar = progressbar.ProgressBar(widgets=self.widgets, maxval=len(sorted_idx))
        bar.start()
        while len(sorted_idx) > 0:
            #Keeping highest pscore bbox as reference.
            rbbox_i = sorted_idx[0]
            #Appending the reference bbox index to filtered list.
            filtered.append(rbbox_i)
            
            #Calculating (xmin,ymin,xmax,ymax) coordinates of all bboxes w.r.t to reference bbox
            overlap_xmins = np.maximum(x_min[rbbox_i],x_min[sorted_idx[1:]])
            overlap_ymins = np.maximum(y_min[rbbox_i],y_min[sorted_idx[1:]])
            overlap_xmaxs = np.minimum(x_max[rbbox_i],x_max[sorted_idx[1:]])
            overlap_ymaxs = np.minimum(y_max[rbbox_i],y_max[sorted_idx[1:]])
            
            #Calculating overlap bbox widths,heights and there by areas.
            overlap_widths = np.maximum(0,(overlap_xmaxs-overlap_xmins+1))
            overlap_heights = np.maximum(0,(overlap_ymaxs-overlap_ymins+1))
            overlap_areas = overlap_widths*overlap_heights
            
            #Calculating IOUs for all bboxes except reference bbox
            ious = overlap_areas/(bbox_areas[rbbox_i]+bbox_areas[sorted_idx[1:]]-overlap_areas)
            
            #select indices for which IOU is greather than threshold
            delete_idx = np.where(ious > threshold)[0]+1
            delete_idx = np.concatenate(([0],delete_idx))
            
            #delete the above indices
            sorted_idx = np.delete(sorted_idx,delete_idx)
            counter += 1
            bar.update(counter)
        #Return filtered bboxes
        return bboxes[filtered].astype('int'), psocres[filtered] #186

    def extract_data(self, image, boxes, scores, threshold, name, save_path):
        """
        extract_data - Function to extract data from the working process

        Input :
            - image : source image 
            - boxes : numpy array with shape [m,x1,y1,x2,y2]
            - scores : Predicted scores for each box
            - threshold : threshold filter float
            - path : folder to save the data

        Output : 
            - Save data in folder given

        ALWAYS THE FIRST CLASS IS THE MAIN OBJECT TO DETECT
        """
        date = datetime.datetime.now()
        timestamp = str(date.day)+str(date.hour)+str(date.minute)
        for idx, box in enumerate(boxes):
            if scores[idx] >= threshold:
                image_substracted = image[box[0]:box[2], box[1]:box[3], :]
                name = "D" + str(idx) + "_" + name + timestamp + ".jpg"
                path = os.path.join(save_path, name)
                cv2.imwrite(path, image_substracted)


    
    def config_file(self, path="./"):
        """
        config_file:
            Reserved function for parameters configuration using yaml files
        
        Input:
            path - location of the configuration yaml file
        """
        with open(os.path.join(path, "config.yml"), 'r') as ymlfile:
            config_file = yaml.load(ymlfile, Loader=yaml.FullLoader)
        postprocessing = config_file['Postprocessing']
        self.radio_mask = postprocessing["radio_mask"]
        self.radio_im = postprocessing["radio_image_fill"]
        self.radio_ext = postprocessing["radio_image_ext"]
        self.iou_threshold = postprocessing["iou_threshold"]
