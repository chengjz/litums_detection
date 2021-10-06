# litums_detection


## 1. Overview

This repository hosts scripts to generate White balanced Cropped litums image and it's RGB value.
<img width="1425" title="overview" alt="Screen Shot 2021-10-06 at 2 21 26 PM" src="https://user-images.githubusercontent.com/36576685/136150870-4f792f41-8037-494c-84da-d67c474fd815.png">


## 2. Methodology
1. Detect QR code by utilizing the tool [QR-Code-Extractor](https://github.com/IdreesInc/QR-Code-Extractor)
2. Apply mask based on QR code location
3. Apply grey world white balance
4. Canny edge detection, smoothing to find black rectangle contours
5. Find strip contours 
6. Locate strip vertices
7. Apply Simple white balance
8. Find litmus paper vertices 
9. Crop largest circle possible in litmus squares
10. Fetch color values

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
#### 3.3.2 processing directory(images)
Change processing_all_img.py, in the line 55 change dirName1 to the directory path in your local host.

Use the following:

```shell
python3 processing_all_img.py
```
