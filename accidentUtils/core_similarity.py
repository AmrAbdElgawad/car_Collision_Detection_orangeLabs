import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob
import os
from core_optical_flow_farneback  import *
import joblib

filename = 'finalized_model.sav'

# load the model from disk
loaded_model = joblib.load(filename)

def similarity(img1,img2):
  # Initiate SIFT detector
  sift = cv2.xfeatures2d.SIFT_create()
  # find the keypoints and descriptors with SIFT
  kp1, des1 = sift.detectAndCompute(img1,None)
  kp2, des2 = sift.detectAndCompute(img2,None)
  # BFMatcher with default params
  bf = cv2.BFMatcher()
  matches = bf.knnMatch(des1,des2, k=2)
  # Apply ratio test
  good = []
  for m,n in matches:
    if m.distance < 0.8*n.distance:
      good.append([m])
  return((len(good)/len(matches))*100)

def crop_object (frame,bbox):
  if bbox[0]>=0 and bbox[1]>= 0  and bbox[2] >=0 and bbox[3] >=0 :
    img=np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cropped_img = img[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
    if cropped_img.shape[0]>=25  or cropped_img.shape[1]>=25 :
      img_name = True
    else:
      cropped_img,img_name=False,False
  else:
    cropped_img,img_name=False,False
  return cropped_img,img_name 

def img_dict(track_id,dict_cars,cropped_img):
  if track_id in dict_cars.keys():
    img1 = dict_cars.get(track_id)[-1]
    if similarity(img1,cropped_img)>8:
      dict_cars.get(track_id).append(cropped_img)
    else:
      pass
  else:
    if len(dict_cars.keys()) ==0:
      dict_cars[track_id]=[]
      dict_cars.get(track_id).append(cropped_img)
    else:
      sim_list=[]
      for i in dict_cars.keys():
        img1 = dict_cars.get(i)[-1]
        sim_list.append([similarity(img1,cropped_img),i])
      max_similarity=max([sublist for sublist in sim_list])
      if max_similarity[0]>20:
        dict_cars.get(max_similarity[1]).append(cropped_img)
      else:
        dict_cars[track_id]=[]
        dict_cars.get(track_id).append(cropped_img)
  return dict_cars


def accident_detection(dict_cars):
  pred_features=None
  pred_list=[]
  for i in dict_cars.keys():
    length= len(dict_cars.get(i))
    if length % 30 == 0:
      # print("30 frame")
      images =dict_cars.get(i)[-30:]
      (h,w)=(200,250)
      resized_images=[]
      for k in images:
        resized_images.append(cv2.resize(k, (w,h), interpolation = cv2.INTER_AREA))
      features=optical_flow_farnebick(resized_images)
      features_mean = features.mean(axis=0)
      pred_features=loaded_model.predict(features_mean.reshape(1,-1))
      pred_list.append([pred_features,i])
  return pred_list



