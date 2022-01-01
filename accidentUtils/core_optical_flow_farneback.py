import cv2
import numpy as np
from numpy import pi as PI
Y_CROP = 45
X_CROP = 15
TRAIN_SIZE = 15000
TEST_SIZE = 4000
number_of_feature_points = 150
number_of_neighbors = 8
FRAME_SIZE = 100
TOTAL_NUMBER_OF_VIDEOS = 246



#preprocess and prepare each video frame
def process_frame(frame):
    if frame is None:
        return None
    height, width, num_of_channels = np.shape(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = frame[Y_CROP: (height - Y_CROP), X_CROP: (width - X_CROP)]
    processed_frame = cv2.resize(frame, dsize=(100, 100))
    return processed_frame


def optical_flow_farnebick(video):

    
    feature_vector = []
    # Take first frame and find corners in it
    old_gray = process_frame(video[0])
    mask = np.zeros_like(old_gray)
    for i in range (1,len(video)):
        frame = video[i]
        if frame is None:
            break
        frame_gray = process_frame(frame)
   
        # calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(old_gray, frame_gray, None,0.5, 3, 15, 3, 5, 1.2, 0)

        #flow=HornSchunck(old_gray, frame_gray)
        # Computes the magnitude and angle of the 2D vectors
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        #magnitude, angle = cv2.cartToPolar(flow[0], flow[1])



        old_gray = frame_gray.copy()
        bin_edges = 18#[i*(PI/6) for i in range(-6, 7)] #bins 336
        magnitude_histogram, bins = np.histogram(magnitude, bins=bin_edges)
        angle_histogram, bins = np.histogram(angle, bins=bin_edges)
        magnitude_histogram = magnitude_histogram / (np.max(magnitude_histogram))
        angle_histogram = angle_histogram / (np.max(angle_histogram))
        totalfeatures=np.hstack((magnitude_histogram,angle_histogram))
        feature_vector.append(totalfeatures)

    return np.array(feature_vector)
  
  
 