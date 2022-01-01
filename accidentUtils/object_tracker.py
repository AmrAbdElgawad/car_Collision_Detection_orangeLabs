import os
import time
import glob
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import core_utils as utils
from core_yolov4 import filter_boxes
from core_config import cfg
from core_similarity import *

import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.python.saved_model import tag_constants
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
import tools_generate_detections as gdet
import pytube
import MediaInfo
#######################################################################
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
########################################################################
# Definition of the parameters
max_cosine_distance = 0.4
nn_budget = None
nms_max_overlap = 1.0

# initialize 
model_filename = 'mars-small128.pb'
framework = 'tf'
model="yolov4"
tiny= False
count= True
info= True
dont_show=True
iou=0.45
score=0.5
input_size = 416
output_format='mp4v'

def load_video(url):
  pytube.YouTube(url).streams.get_highest_resolution().download("/content/project/videos")
  path = os.listdir("/content/project/videos")[0]
  return "/content/project/videos/"+path


def accidentDetection(video_path,output,saved_model_loaded):

    #X=load_video(video_path)
    #video_path=X

    dict_cars=dict()
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True

    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config()
    #
    # otherwise load standard tensorflow saved mod
    #saved_model_loaded = tf.saved_model.load(weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None


    # get video ready to save locally
    if output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*output_format)
        out = cv2.VideoWriter(output, codec, fps, (width, height))
    frame_num = 0
    #skip first 10 frames
    start_frame_number = 10
    vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)

    # while video is running
    numacc=0
    numnotacc=0 
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num +=1
       # print('Frame #: ', frame_num)
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # run detections on tflite 
        if framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if model == 'yolov3' and tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=iou,
            score_threshold=score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['car','truck']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            # print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]


        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            
            if info:
                # print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
                cropped_img,img_name= crop_object (frame,bbox)
                if img_name == False:
                  continue
                # save image
                dict_cars =img_dict(track.track_id,dict_cars,cropped_img) 
            
        list_pred= accident_detection(dict_cars)
        if list_pred==None:
          pred = "There is no Accident detected"
          cv2.putText(frame,pred,(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        
        for inner_list in list_pred:
          if inner_list[0]==0:
            acc="No Accident "
            numacc+=1
          elif inner_list[0]==1:
            acc="Accident"
            numnotacc+=1
          else:
            acc="None"

          pred="object : "+str(inner_list[1])+ " is predicted : "+acc
          cv2.putText(frame,pred,(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        # print(list_pred)
                
            
                  
                
            
        # draw bbox on screen
        for track in tracker.tracks:
          if not track.is_confirmed() or track.time_since_update > 1:
                continue 
          bbox = track.to_tlbr()
          class_name = track.get_class()
          color = colors[int(track.track_id) % len(colors)]
          color = [i * 255 for i in color]
          cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
          cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
          cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            
        

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        # print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
       
        if not dont_show:
            cv2.imshow("Output Video", result)
        
        # if output flag is set, save video file
        if output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
    if numacc==0:
      return "There is no accident occured in the video"
    if numnotacc/numacc>=3:
      return "There is no accident occured in the video"
    else:
      return "An accident has occured in the video"
          
    cv2.destroyAllWindows()