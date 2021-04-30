## litums_detection

------

### 1. Overview

------

This repository hosts scripts to generate White balanced Cropped litums image and it's RGB value.

------


### 1.1. Prerequisites

If you don't have Git, install it from [Git Downloads](https://git-scm.com/downloads).
And Python3 is also required.

------

### 2. Setting Environment

------

#### 2.1. Install/Upgrade virtualenv

On macOS and Linux:

```shell
pip install virtualenv==1.7.1.2
```

#### 2.2. Create virtualenv 

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

#### 2.3. Clone this repository

Use the following:

```shell
git clone https://github.com/chengjz/litums_detection.git
```

#### 2.4. Install packages
cd to this directory and install packesges
Use the following:

```shell
cd litums_detection
```

```shell
pip3 install -r requirements.txt
```

### 3. Use

------

#### 3.1. processing single image
Use the following:

```shell
python3 test-single-image.py 
```
#### 3.1. processing directory(images)
Change processing_all_img.py, in the line 55 change dirName1 to the directory path in your local host.

Use the following:

```shell
python3 processing_all_img.py
```
