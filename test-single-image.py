
import cv2 
import math
import numpy as np
import sys
import glob
import os
import litums_detection as ld
import matplotlib.pyplot as plt

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

frame = cv2.imread('test.jpg')
lower_crop_litmus_img, upper_crop_litmus_img = ld.get_upper_lower_litums(frame)
cv2.imwrite("lower_crop_litmus_img.jpg", lower_crop_litmus_img)
cv2.imwrite("upper_crop_litmus_img.jpg", upper_crop_litmus_img)

lr, lg, lb = getAverageRGBN(lower_crop_litmus_img)
hr, hg, hb = getAverageRGBN(upper_crop_litmus_img)

print("total R:" + str(lr) + "total G:" + str(lg) +  "total B:" + str(lb) + "total Sum:" + str(lr + lg + lb))
print("free R:" + str(hr) + "free G:" + str(hg) + "free B:" + str(hb) + "free Sum:" + str(hr + hg + hb))
