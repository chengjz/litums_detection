#!/usr/bin/env python
import cv2
import litmus_detection as ld
import math
import numpy as np
import pandas as pd
import os
import shutil


# make folders and replace with relevant paths if you are doing this on local computer
#test_path = "/mnt/c/Users/.../test" etc.
#output_path = 
#manual_path = 
#output_xlsx_path = 


def process_all_images(test_path, output_path, manual_path, output_xlsx_path):
    # creating the free and total dataframes that will be filled with bgr values
    free_df = pd.DataFrame({'filename': [],'sample ID' : [],'blue val': [],'green val': [],'red val': []})
    total_df = pd.DataFrame({'filename': [],'sample ID' : [],'blue val': [],'green val': [],'red val': []})
    index = 0

    # iterating through all of the images in the testing photos folder
    for root, dirs, files in os.walk(test_path):
        for fn in files:
            if fn.find(".jpg") > 0 or fn.find(".png") > 0 or fn.find(".jpeg") > 0:
                if fn.find(".jpeg") > 0:
                    name = fn[:-5]
                if fn.find(".jpg") > 0 or fn.find(".png") > 0:
                    name =  fn[:-4]

                file_Path = os.path.join(root, fn)
                try:
                  frame = cv2.imread(file_Path)
                  # calling the litmus detection script
                  free_circle_cropped_img, free_bgr, total_circle_cropped_img, total_bgr, wb = ld.processing_img(frame)
                  print(name)
                  print("free blue, green, red = ",free_bgr)
                  print("total blue, green, red = ",total_bgr)
                  # outputing the wb strip and detected litmus squares to the cropped photos output folder
                  cv2.imwrite(os.path.join(output_path, name + "_free.jpg"), free_circle_cropped_img)
                  cv2.imwrite(os.path.join(output_path, name + "_total.jpg"), total_circle_cropped_img)
                  cv2.imwrite(os.path.join(output_path, name + "_wb_strip.jpg"), wb)
                  
                  # adding the concentration and b, g and r values to the pandas dataframes. Note that fn[:4] assumes the naming convention used
                  # labels the concentration at the start of the filename i.e. 1.00_image.jpg but this can be changed
                  free_df.loc[index] = [fn] + [fn[:4]] + free_bgr
                  total_df.loc[index] = [fn] + [fn[:4]] + total_bgr
                  index += 1
                  print("----------------------------------------------------------")
                except:
                  #if an error occurs (such as QR code not being read)
                  print(fn)
                  print("error")
                  shutil.move(file_Path , os.path.join(manual_path,fn))
                  print("----------------------------------------------------------")

    # printing the final free and total dataframes and outputting it to excel   
    print("free data frame")
    print(free_df)
    print("total data frame")
    print(total_df)

    if os.path.exists(output_xlsx_path):
        with pd.ExcelWriter(output_xlsx_path,engine="openpyxl",mode="a",if_sheet_exists="new") as writer:
            free_df.to_excel(writer,sheet_name="free")
            total_df.to_excel(writer,sheet_name="total")
    else:
        with pd.ExcelWriter(output_xlsx_path,mode="w") as writer:
            free_df.to_excel(writer,sheet_name="free")
            total_df.to_excel(writer,sheet_name="total")


#process_all_images(test_path, output_path, manual_path, output_xlsx_path)

