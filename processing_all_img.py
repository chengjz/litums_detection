
import cv2 
import math
import numpy as np
import sys
import glob
import os
import litums_detection as ld
import matplotlib.pyplot as plt

target_dir1 = "Black Background"
target_dir2 = "black_background"


def averagePixels(img):
      r, g, b = 0, 0, 0
      count = 0
      for x in range(img.size[0]):
          for y in range(img.size[1]):
              tempr,tempg,tempb = img[x,y]
              r += tempr
              g += tempg
              b += tempb
              count += 1
      # calculate averages
      return (r/count), (g/count), (b/count), count


def getAverageRGBN(image):
  """
  Given PIL Image, return average value of color as (r, g, b)
  """
  # get image as numpy array
  im = np.array(image)
  # get shape
  w,h,d = im.shape
  # change shape
  im.shape = (w*h, d)
  # get average
  return tuple(np.average(im, axis=0))

# global total_img 
# global error
def get_all_img():
    total_img = 0
    error = 0
    # img_dir = "/Users/jasoncheng/Desktop/cv_capstone/Chlorine Residual Project/Photos/Clorox Photos/Titrated/" # Enter Directory of all images 
    # data_path = os.path.join(img_dir,'*png')
    # # files = glob.glob(data_path)
    # data = []
    # files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]

    # Get the list of all files in directory tree at given path
    imgs = [] #Project/03_Photos\ 2/Leah\'s\ iPhone\ 5/Wet/unedited/outside/black_background/1.1.JPG 
    dirName1 = "/Users/jasoncheng/Desktop/cv_capstone/Chlorine Residual Project/03_Photos 2/Alyssa's iPhone 8/Wet/Unedited/Laboratory"
    dirName = "/Users/jasoncheng/Desktop/cv_capstone/Chlorine Residual Project/03_Photos 2" # Enter Directory of all images 
    subpath = "/Users/jasoncheng/Desktop/cv_capstone/Chlorine Residual Project/03_Photos/1.99 Edits/iPhone 5/HF1.99.png"
    img = cv2.imread(subpath)
    cv2.imwrite("test.jpg", img)

    low_conc_rgb = {}
    upper_conc_rgb = {}
    # listOfFiles = list()

    listofDirs = list()
    for (dirpath, dirnames, filenames) in os.walk(dirName1):
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
                total_img += 1
                if fn.find("_crop_litmus_img.jpg") > 0 or fn.find("_lower.jpg") > 0 or fn.find("_upper.jpg") > 0:
                    continue
                file_Path = os.path.join(dir_name, fn)
                print(file_Path)

                base = os.path.basename(file_Path)
                conc = os.path.splitext(base)[0]
                print("conc")
                print(conc)
                try:
                    frame = cv2.imread(file_Path)
                    lower_crop_litmus_img, upper_crop_litmus_img = ld.get_upper_lower_litums(frame)

                    cv2.imwrite(os.path.splitext(file_Path)[0] + "_lower.jpg", lower_crop_litmus_img)
                    cv2.imwrite(os.path.splitext(file_Path)[0] + "_upper.jpg", upper_crop_litmus_img)
                    lr, lg, lb = getAverageRGBN(lower_crop_litmus_img)
                    hr, hg, hb = getAverageRGBN(upper_crop_litmus_img)
                    if low_conc_rgb.get(conc) is None:
                        # low_conc_rgb[conc] = [lr, lg, lb, lr + lg + lb]
                        low_conc_rgb[conc] = [lr + lg + lb]
                        # upper_conc_rgb[conc] = [hr, hg, hb, hr + hg + hb]
                        upper_conc_rgb[conc] = [hr + hg + hb]
                    else:
                        # low_conc_rgb[conc] += [lr, lg, lb, lr + lg + lb]
                        # upper_conc_rgb[conc] += [hr, hg, hb, hr + hg + hb]

                        low_conc_rgb[conc] += [lr + lg + lb]
                        upper_conc_rgb[conc] += [hr + hg + hb]

                    print(conc, low_conc_rgb[conc], upper_conc_rgb[conc])
                except:
                    error += 1
                    print("-------------------")
                    print(dir_name)
                    print(fn)
    return total_img, error, low_conc_rgb, upper_conc_rgb


total_img, error, low_conc_rgb, upper_conc_rgb = get_all_img()
# print(total_img)
# print(error)
# print(upper_conc_rgb)

# low_conc_rgb = {'0.00': [587.6304347826086], '0.16': [590.428071928072], '0.13': [599.0602836879433], '1.1': [217.2965156794425], '1.37': [421.2183333333333], '0.49': [532.1217750257998], '0.61': [501.05838739573676], '0.98': [458.5176136363636], '0.530': [568.030519005848], '1.02': [698.8311170212767], '0.310': [573.1159297052154], '0.8': [477.45690789473684], '1.10': [493.1226708074534], '2.18': [238.6639231824417], '0.36': [567.8944128787879], '1.99': [548.8884943181818], '0.000': [623.8737927292458], '0': [633.5995397008055]}
for key in upper_conc_rgb:

    plt.scatter(key, upper_conc_rgb[key])

plt.legend()
plt.show()

# colors = list("rgbcmyk")

# for data_dict in low_conc_rgb.values():
#    x = data_dict.keys()
#    y = data_dict.values()
#    plt.scatter(x,y,color=colors.pop())

# plt.legend(low_conc_rgb.keys())
# plt.show()

# for index in low_conc_rgb.keys():
#     for rgb in low_conc_rgb.get(index):
#         plt.plot(index, rgb, '.')
#         plt.xlabel("RGB Sum")
#         plt.ylabel("Chlorine")
#         plt.yticks([0,1,2])
#         plt.xticks([500,600,700])

# plt.ylabel("Chlorine")
# plt.subplot(2, 3, ctr)
# plt.title(index)
# plt.plot(data,conc[index], '.')
# plt.xlabel("RGB Sum")
# plt.ylabel("Chlorine")
# plt.yticks([0,1,2])
# plt.xticks([500,600,700])
# plt.subplot(2, 3, ctr+3)
# plt.plot(basis["r"],conc[index], '.', color="r")
# plt.plot(basis["g"],conc[index], '.', color="g")
# plt.plot(basis["b"],conc[index], '.', color="b")
# plt.ylabel("Chlorine")
# plt.xlabel("RGB Count")
# plt.yticks([0,1,2])
# plt.xticks([100, 150,200,250])
# ctr += 1

# conc_rgb = {}
# frame = cv2.imread('upper_crop_litmus_img.jpg')
# lr, lg, lb = getAverageRGBN(frame)
# print(lr, lg, lb)
# print(sum(getAverageRGBN(frame)))
# if conc_rgb.get('conc') is None:
#     conc_rgb['conc'] = [[lr, lg, lb, sum(getAverageRGBN(frame))]]
# else:
#     conc_rgb['conc'] += [[lr, lg, lb, sum(getAverageRGBN(frame))]]
# print(conc_rgb)