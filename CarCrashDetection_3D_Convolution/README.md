
 

<h1 color="green"><b>Car Crash Detection Package</b></h1>

---

<h1 color="green"><b>Abstract</b></h1>
<p>Car Crash Detection can be seen as the Detection of Accident or Not in a video according to the actions occurring in it. It has
become one of the most challenging and attractive problems in video classification and detection fields.
The problem itself is difficult to solve by traditional video processing methods because of several challenges such as
the background noise, sizes of subjects in different videos, and the speed of Cars.Derived from the progress of
deep learning methods, several directions are developed for video detection, such as the
long-short-term memory (LSTM)-based model, two-stream convolutional neural network (CNN) model, and the convolutional 3D model.
Car Crash Detection is used in some surveillance systems and video processing tools.
Our main problem is Accident Detection which we achieved to solve by using transfer learning on pretrained convolutional 3D models
that aim to recognize the motions and actions of Cars.
All models use Kinetics-400 dataset for the pretrained part and Vision-based Accident Detection From Surveillance Cameras dataset
for the finetuned part.</p>


<h1 color="green"><b>Pytorch Pretrained Models</b></h1>
<p>All pretrained models can be found in this link.
 <a href="https://pytorch.org/vision/stable/models.html">lhttps://pytorch.org/vision/stable/models.html</a></p>


<h1 color="green"><b>Instructions to Install our Car Crash Detection Package</b></h1>


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
CrashUtils.crashDetection(inputPath,seq,skip,outputPath,showInfo=False,thresholding=0.85)
```
4. Show the Output Video with Detection:

```python
from moviepy.editor import *
VideoFileClip(outputPath, audio=False, target_resolution=(300,None)).ipython_display()
```
5. To Start Detect the Accident on Streaming

```python
CrashUtils.start_streaming(streamingURL,thresholding=0.85)
```