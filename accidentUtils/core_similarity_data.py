import numpy as np
import cv2
from matplotlib import pyplot as plt



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

  


