
import cv2 
import math
import numpy as np
import sys
import glob
import os
import litums_detection as ld

target_dir1 = "Black Background"
target_dir2 = "black_background"

def get_all_img():
    # img_dir = "/Users/jasoncheng/Desktop/cv_capstone/Chlorine Residual Project/Photos/Clorox Photos/Titrated/" # Enter Directory of all images 
    # data_path = os.path.join(img_dir,'*png')
    # # files = glob.glob(data_path)
    # data = []
    # files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]

    # Get the list of all files in directory tree at given path
    imgs = []
    dirName1 = "/Users/jasoncheng/Desktop/cv_capstone/Chlorine Residual Project/03_Photos 2/LG flip phone/wet/unedited/lab"
    dirName = "/Users/jasoncheng/Desktop/cv_capstone/Chlorine Residual Project/03_Photos 2" # Enter Directory of all images 
    subpath = "/Users/jasoncheng/Desktop/cv_capstone/Chlorine Residual Project/03_Photos/1.99 Edits/iPhone 5/HF1.99.png"
    img = cv2.imread(subpath)
    cv2.imwrite("test.jpg", img)

    # listOfFiles = list()

    listofDirs = list()
    for (dirpath, dirnames, filenames) in os.walk(dirName):
        if target_dir1 in dirnames:
            print("-------------------")
            print("dirpath")
            print(dirpath)
            print(dirnames)
            print(filenames)
            listofDirs += [os.path.join(dirpath, target_dir1)]
        elif target_dir2 in dirnames:
            print("-------------------")
            print("dirpath")
            print(dirpath)
            print(dirnames)
            print(filenames)
            listofDirs += [os.path.join(dirpath, target_dir2)]
            # listOfFiles += [os.path.join(dirpath, file) for file in filenames]

    print(len(listofDirs))
    for dir_name in listofDirs:
        for (dirpath, dirnames, filenames) in os.walk(dir_name):
            # print("-------------------")
            # print("dirpath")
            # print(dir_name)
            # print(dirpath)
            # print(dirnames)
            # print(filenames)
            listOfFiles = list()
            for fn in filenames:
                if fn.find("_crop_litmus_img.jpg") > 0 or fn.find("_lower.jpg") > 0 or fn.find("_upper.jpg") > 0:
                    continue
                file_Path = os.path.join(dir_name, fn)
                print(file_Path)
                try:
                    frame = cv2.imread(file_Path)
                    lower_crop_litmus_img, upper_crop_litmus_img = ld.get_upper_lower_litums(frame)
                    cv2.imwrite(os.path.splitext(file_Path)[0] + "_lower.jpg", lower_crop_litmus_img)
                    cv2.imwrite(os.path.splitext(file_Path)[0] + "_upper.jpg", upper_crop_litmus_img)
                except:
                    print("-------------------")
                    print(dir_name)
                    print(fn)

get_all_img()