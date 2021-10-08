# litums_detection


## 1. Overview

This repository hosts scripts to generate White balanced Cropped litums image and it's RGB value.
<img width="1425" title="overview" alt="Screen Shot 2021-10-06 at 2 21 26 PM" src="https://user-images.githubusercontent.com/36576685/136150870-4f792f41-8037-494c-84da-d67c474fd815.png">


## 2. Methodology
1. Apply white balance to improve the performance of edge detection and qr-code-extractor
2. Detect the qr-code location by utilizing the tool QR-Code-Extractor and then rotate the img based on the qr-code angle. For the detailed description of the qr-code-extractor, refers here: https://github.com/IdreesInc/QR-Code-Extractor#methodology---how-it-works 
3. Find exact position of the black background
    1. Based on the QR-Code Location, Get the approximate position of the black background to narrow down the search area
    2. Masked the trivial part
    3. Get the exact position of the black background based on the approximate position and QR-Code area

4. Get the strip position with the following steps:
* Find the major part of the strip:          
    1. use the canny edge detection to find all the contours
    2. Set the (back_ground_area * fixed_ratio) as the upper bound of the contours area
    3. traverse each contour, calculate the bounding area of it’s minAreaRect
    4. the contour of major part should be the contour with maximum bounding area under the upper bound
* Find the minor parts of the strip:
    1. Build a virtual box by extending the edges of the rectangle containing major part of strip
    2. Traverse each contour, if the contour meets the following criteria, we assume it's also part of the strip, concatenate this contour to the contour of majority_strip :
        (a) The contour the moment of the contour is inside the virtual box
        (b) (edge point to the virtual rectangle is inside the virtual box
        or the distance of edge point to the virtual rectangle within tolerance)
* The rationale behind this step: canny edge detection may detect the strip as separated parts and the morphological transformations is not enough to group the separated contours of the strip
5. Apply simple_white_balance to the strip
    * white_balanced based on the white pixel of the strip. For each RBG channel, find the pixel's peak value, and then extend the value range of the strip from [0, peak value] to [0, 255]. 
    * The reason behind it is that we already know that the RGB value of the strip white part should be (255, 255, 255)
6. Get litmus square: Given the strip img and the vertices of the strip rectangle, return the exact position of the litmus with some math. 
7. Get crop circle of the litmus: Given a litmus, find the biggest circle inside this rectangle
8. Get rgb value of the cropped circle

## 3. How to Use this script

### 3.1 Prerequisites

If you don't have Git, install it from [Git Downloads](https://git-scm.com/downloads).
And Python3/pip3 is also required.


### 3.2 Setting Environment


#### 3.2.1 Install/Upgrade virtualenv

On macOS and Linux:

```shell
pip3 install virtualenv
```

#### 3.2.2 Create virtualenv 

On macOS and Linux:

```shell
python3 -m venv tutorial-env
```
Once you’ve created a virtual environment, you may activate it.

On Windows, run:

```shell
tutorial-env\Scripts\activate.bat
```

On Unix or MacOS, run:

```shell
source tutorial-env/bin/activate
```

if you want to deactivate this virtualenv, run the following:
```shell
deactivate
```

#### 3.2.3 Clone this repository

Use the following:

```shell
git clone https://github.com/chengjz/litums_detection.git
```

#### 3.2.4 Install packages
cd to this directory and install packesges
Use the following:

```shell
cd litums_detection
```

```shell
pip3 install -r requirements.txt
```



### 3.3 Execute

#### 3.3.1 processing single image
Use the following:

```shell
python3 test-single-image.py 
```
