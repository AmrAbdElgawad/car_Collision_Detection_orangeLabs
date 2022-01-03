# Car_Collision_Detection_OrangeLabs
# this is car collision detection system me and my team created it as graduation project in ITI ai-pro track undersupervision and mentoring of orange labs
## Car Crash Detection can be seen as the Detection of Accident or Not in a video according to the actions occurring in it. It has become one of the most challenging and attractive problems in video classification and detection fields.
## The problem itself is difficult to solve by traditional video processing methods because of several challenges such as the background noise, sizes of subjects in different videos, and the speed of Cars.Derived from the progress of deep learning methods, several directions are developed for video detection, such as the violent flow , the long-short-term memory (LSTM)-based model, two-stream convolutional neural network (CNN) model, and the convolutional 3D model. Car Crash Detection is used in some surveillance systems and video processing tools.
# Our main problem is Accident Detection which we achieved to solve by using 2 aprroches `VIF` and `3D Convolution`.</p>
## Our Pipeline for VIF :
### read the camera stream then run our segmentation methodology of each car in the scene then compute VIF for each tracked object and give each feature vector of segmented cars to svm to detect whether this object has collision or not .
![alt text](https://github.com/AmrAbdElgawad/car_Collision_Detection_orangeLabs/blob/main/images/pipeline.png)
### - We use Yolo V4 to detect all vehicles in the scene 
### - track each detected vehicle using deep sort
### - use second step verification using our sift based similarity method to correct tracking errors
### - create trajectory of each car and segment it into smaller videos as shown in the image below
![alt text](https://github.com/AmrAbdElgawad/car_Collision_Detection_orangeLabs/blob/main/images/cars_split.png)
### - compute Violent flow for each segmented object 
### - get the magnitude and angle histogram and combine them into 36  bins
### - get the mean of these values over 30 frames and give the results to SVM
## the main file to run the project is our colab kernel to run the demo 

## Our Pipeline for 3D Convolution (3D_ResNet-18):
### we used transfer learning on pretrained convolutional 3D models that aim to recognize the motions and actions of Cars and all models use Kinetics-400 dataset for the pretrained part and Vision-based Accident Detection From Surveillance Cameras dataset for the finetuned part.

![alt text](https://github.com/AmrAbdElgawad/car_Collision_Detection_orangeLabs/blob/main/images/1.jpg)

<h1 color="green"><b>Our 3D_Convolution Pipeline</b></h1>

![alt text](https://github.com/AmrAbdElgawad/car_Collision_Detection_orangeLabs/blob/main/images/2.jpg)

<h1 color="green"><b>Pytorch Pretrained Models</b></h1>
<p>All pretrained models can be found in this link.
 <a href="https://pytorch.org/vision/stable/models.html">https://pytorch.org/vision/stable/models.html</a></p>
 
<h1 color="green"><b>Instructions to Install our Car Crash Detection Package</b></h1>
<p>Pip Package can be found in this link.
 <a href="https://pypi.org/project/Car-Crash-Detection/">https://pypi.org/project/Car-Crash-Detection/</a></p>

1. Install:

```python
pip install Car-Crash-Detection
pip install pytube
```

2. Download the Finetunned Model Weights

```python
import gdown
url = 'https://drive.google.com/uc?id=1-8TyT7MkAS7LLsRTbO03tDsuuoBM6q1D'
model = 'model_ft.pth'
gdown.download(url, model, quiet=False)
```
3. Detect Accident or Not by Pass your Local Video:

```python
from car_crash_detection import CrashUtils
# Run the Below Function by Input your Test Video Path to get the outPut Video with Accident Detection or Not
CrashUtils.crashDetection(inputPath,seq,skip,outputPath,showInfo=False,thresholding=0.75)
```
4. Show the Output Video with Detection:

```python
from moviepy.editor import *
VideoFileClip(outputPath, audio=False, target_resolution=(300,None)).ipython_display()
```
5. To Start Detect the Accident on Streaming

```python
CrashUtils.start_streaming(streamingURL,thresholding=0.75)
```
