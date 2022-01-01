# car_Collision_Detection_orangeLabs
# this is car collision detection system me and my team created it as graduation project in ITI ai-pro track undersupervision and mentoring of orange labs
## Our Pipeline :
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
