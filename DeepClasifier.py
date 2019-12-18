#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 11:12:43 2019

@author: DavidFelipe
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import yaml
import numpy as np
import cv2
import progressbar
import DataGenerate
print("TensorFlow version is ", tf.__version__)

class DeepClasifier:
    def __init__(self):
        self.config_file()
        self.IMG_SHAPE = (self.image_size, self.image_size, 3)
        self.model = []
        self.model_flag = False
        self.vector_boxes = [np.array([0,0,0,0,0])]
        self.vector_boxes_final = np.array([0,0,0,0,0])
        config = tf.compat.v1.ConfigProto( device_count = {'GPU':1, 'CPU':60} )
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(self.sess)
        
    def prepare_list(self, list_images, size = 1000):
        """
        prepare_list - Function to divide the dataset into batches with given size
        
        Input:
            list_images - list of the dataset
            size - size of the batches
        
        Return:
            list_batches - list of the dataset divided in batches
        """
        batch = []
        list_batches = []
        counter = 0
        for item, img in enumerate(list_images):
            batch.append(img)
            counter += 1
            if counter >= size:
                counter = 0
                list_batches.append(batch)
                batch = []
            elif item == len(list_images) - 1:
                list_batches.append(batch)
        return list_batches
        
    def check_detector(self, point, box, score):
        """
        Input : 
            point - tupla (x,y) point of evaluation
            box - size (w,h) of the box evaluation
            score - Deep learning model score classification
        
        Output :
            flag - saved the position
        """
        if self.deep_classes_num == 3:
            fail_parameter = score[1] + score[2]
        else :
            fail_parameter = score[1]
            
        if score[0] > self.score_min_Acce and fail_parameter < self.score_min_Fail:
            line = np.array([point[0], point[1], box[0], box[1], score[0]])
            self.vector_boxes = np.vstack((self.vector_boxes, line)) 
       
    def extract_part(self, image, point):
        
        point = point.astype(np.uint16)
        limits = (point[2]+point[1], point[3]+point[0])
        cv2.rectangle(image, (point[1],point[0]), limits, (255,0,0), 1)
        cv2.imshow("box1",image)
                
    def refresh(self):
        self.vector_boxes = [np.array([0,0,0,0,0])]
#        self.vector_boxes_final = np.array([0,0,0,0,0])
    
    def predict_batches(self, list_batches):
        """
        Predict_batches - Funntion to predict multiple batches
        Input:
            list_batches - batch list of divided dataset
        Output:
            results_batches - np.array of the results of the prediction
        """
        widgets = [progressbar.Percentage(),
            ' ', progressbar.Bar(),
            ' ', progressbar.ETA()]
        results_batches = np.array([]) # 3 zeros because is the output of the cnn 3 classes
        bar = progressbar.ProgressBar(widgets=widgets, maxval=len(list_batches)-1)
        bar.start()
        for item, batch in enumerate(list_batches):
            result = self.predict_batch(batch)
            if item == 0:
                results_batches = result.copy()
            else:
                results_batches = np.vstack((results_batches, result))
            bar.update(item)
        bar.update(len(list_batches)-1)
        return results_batches
    
    def predict_batch(self, list_img):
        batch_size = len(list_img)
        batch_holder = np.zeros((batch_size, self.image_size, self.image_size, 3))
        for j, img in enumerate(list_img):
            temp = cv2.resize(img.astype("uint8"),(self.image_size, self.image_size))
            temp = image.img_to_array(temp)
            temp = np.expand_dims(temp, axis=0)
            temp = keras.applications.mobilenet.preprocess_input(temp)
            batch_holder[j, :] = temp[0]
        results = self.model.predict_on_batch(batch_holder)
        return results
        
    def predict_list(self, list_img):
        predict_list = []
        for img in list_img:
            Image = cv2.resize(img.astype("uint8"),(self.image_size, self.image_size))
            Image = image.img_to_array(Image)
            img_array_expanded_dims = np.expand_dims(Image, axis=0)
            target = keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)
            predict_list.append(target)
        results = self.model.predict_on_batch(predict_list)
        return results
            
    def predict(self, img):
        if(self.model_flag):
            try:
                Image = cv2.resize(img.astype("uint8"),(self.image_size, self.image_size))
                Image = image.img_to_array(Image)
                img_array_expanded_dims = np.expand_dims(Image, axis=0)
                target = keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)
                results = self.model.predict(target)
                return results
            except:
                return 0
    
    def classification_point(self, images, anchor_boxes):
        """
        Input :
            images - images extracted by the anchor boxes
            anchor_boxes - dimensions of objects searched
        
        Output :
            Score_mean - mean of all the anchor boxes classification
            Score - Score of the best anchor box
            Anchor_box - Anchor box of the best score
        """
        flag = False
        if len(images)>0:
            score_array = np.array([0,0,0])
            mean_score = score_array.copy()
            for item, image_item in enumerate(images):
#                if(type(image_item) != int):
                score = self.predict(image_item)
                if(type(score) != int):
                    score_array = np.vstack((score_array, score[0]))
#            if len(score_array) > 1:
            score_array = np.delete(score_array,[0],axis=0)
            mean_score = score_array.mean(axis=0)
#                mean_fail = 1 + self.score_min_Fail
#                if len(mean_score) == 3:
            mean_fail = mean_score[1] + mean_score[2]
            if (mean_score[0] > self.score_min_Acce and mean_fail < self.score_min_Fail):
                index = score_array.argmax(axis=0)
                box = anchor_boxes[index[0]]        
                image_scored = images[index[0]]
                flag = True
                return flag, image_scored, box , score_array[index[0]][0]
            else:
                return flag, -1, -1, -1
        else:
            return flag, -2, -2, -2
            
    def reclasification(self, img):
        flag = False
        datachange = DataGenerate.DataAugmentation("","",tools=False)
        flip_img, _ = datachange.flip(img,"",full=False)
        rotate_img, _ = datachange.rotate(img,"",full=False)
        block = [flip_img[0], rotate_img[0], rotate_img[1]]
        predict_block = np.array([0,0,0]) ## Dimension class quantify
        for image_block in block:
            results = self.predict(image_block)
            line = np.copy(results[0])
            predict_block = np.vstack((predict_block, line))
        predict_block = np.delete(predict_block, 0, 0)
        mean_score = predict_block.mean(axis=0)
#        print(mean_score)
        mean_fail = mean_score[1] + mean_score[2] 
        if (mean_score[0] > self.score_min_Acce_R and mean_fail < self.score_min_Fail_R):
            flag = True
        return flag
    
    def load_model(self):
        try:
            self.model = tf.keras.models.load_model(self.path_model+self.model_h5)
            self.model_flag = True
            #self.model.summary()
            print(" &&& Model Loaded")
        except:
            print("Problems importing the model ... %%")
        
    def create_model(self):
        # Create the base model from the pre-trained model MobileNet V2
        base_model = tf.keras.applications.MobileNetV2(input_shape=self.IMG_SHAPE, include_top=False, weights=None)
        self. model = tf.keras.Sequential([
          base_model,
          keras.layers.GlobalAveragePooling2D(),
          keras.layers.Dense(2, activation=tf.nn.softmax)
        ])
        self.model_flag = True
        
    def load_weights(self):
        if(self.model_flag):
            self.model.load_weights(self.path_weights+self.weights)
            try:
                print(self.path_weights+self.weights)
                
                self.model.summary()
            except:
                self.model_flag = False
                print("NOT FOUND THE MODEL FILE .h5 ... %%")

    def compile_model(self):
        if(self.model_flag):
            self.model.compile(optimizer='adam',
                               loss='sparse_categorical_crossentropy',
                               metrics=['accuracy'])
        else:
            print("No model created ... %%")
        
    def config_file(self, path="./"):
        with open(os.path.join(path, "config.yml"), 'r') as ymlfile:
            config_file = yaml.load(ymlfile, Loader=yaml.FullLoader)
        deep_conf = config_file['deepclasifier']
        self.image_size = deep_conf["image_size"]
        self.path_weights = deep_conf["path_weights"]
        self.weights = deep_conf["weights"]
        self.path_model = deep_conf["path_model"]
        self.model_h5 = deep_conf["model"]
        self.score_min_Acce = deep_conf["score_min_Acce"]
        self.score_min_Fail = deep_conf["score_min_Fail"]
        self.score_min_Acce_R = deep_conf["score_min_Acce_R"]
        self.score_min_Fail_R = deep_conf["score_min_Fail_R"]
        self.iou_threshold = deep_conf["iou_object"]
        self.deep_classes_num = deep_conf["deep_classes_num"]
