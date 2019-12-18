#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 11:20:16 2019

@author: DavidFelipe
"""
try:
    import progressbar
    import SubsetCut
    import SoftCut
    import GuiCut
    import DeepClasifier
    from Postprocessing import Postprocessing
    from Report import Report
    import time
except:
    print(" PLEASE REVIEW THE MODULES THAT NEEDS THE SOFTWARE - AN ERROR WAS OCCURRED")
    
widgets = [progressbar.Percentage(),
            ' ', progressbar.Bar(),
            ' ', progressbar.ETA()]

"""
initialization
"""
manager = Report()
paths = manager.config_file()
image, name = manager.load_image(paths)
if type(name) == int:
    print("Not found valid format to start the process ... ")
    time.sleep(3)
    exit()

"""
cut procedure
"""
subsetcut = SubsetCut.subsetcut(image.copy(), fit=True)
subsetcut.run()
image_segmented = subsetcut.masked_image.copy()

"""
Tool definitions
"""
guicut = GuiCut.GuiCut(image_segmented.copy(), divide=500, classes=2)
softcut = SoftCut.SoftCut(image_segmented.copy(), verbose=0)
deepclasifier = DeepClasifier.DeepClasifier()
deepclasifier.load_model()#    exit()

"""
Configurations
"""
anchor_boxes, mean_size = guicut.run()
points_grid, image_grid = softcut.get_grid(distance_points=10, flag_output=1)

"""
Clasification point
"""
bar = progressbar.ProgressBar(widgets=widgets, maxval=len(points_grid)-1)
bar.start()
deepclasifier.refresh()
image_time = time.time()
co = 0
for item, point in enumerate(points_grid):
    try:
        curr = time.time()
        subset_images = softcut.windows_extract(point, anchor_boxes)
        """
        Clasification point 
        """
        if len(subset_images) > 0:
            if subset_images[0].any() != 0:
                co += 1
                flag_cfp, subset_image, box, score = deepclasifier.classification_point(subset_images, anchor_boxes)
                if flag_cfp:
                    """
                    Reclasification point
                    """
                    flag_reclas = deepclasifier.reclasification(subset_image)
                    if flag_reclas:
                        """
                        Object Check Detector
                        """
                        deepclasifier.check_detector(point, box, score)
        bar.update(item)
        print(time.time() - curr)
    except KeyboardInterrupt:
        break
    
Total_time = time.time() - image_time

"""
Postprocessing
"""
finalprocess = Postprocessing(image.copy(), deepclasifier.vector_boxes)
count = finalprocess.counter
pointed_image = finalprocess.container[0]
boxed_image = finalprocess.image_drawed[0]

"""
Generate report
"""
manager.Generate(name, paths[1], Total_time, count, boxed_image, pointed_image)
manager.Generate_csv(deepclasifier.vector_boxes, name)
finish = input("Push any key to finish ")
#exit()

