# For Youtube Download.
import io 
from pytube import YouTube
from IPython.display import HTML
from base64 import b64encode


import os
import cv2
import time
import copy
import glob
import torch
import gdown
import argparse
import statistics
import threading
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
from moviepy.editor import *
import albumentations as A
from collections import deque
# from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DATASET_DIR = ""
CLASSES_LIST = ['accident','no_accident']
SEQUENCE_LENGTH = 15
batch_size=4
predicted_class_name = ""


output = '.\\model_ft.pth'
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(output)))
modelWeights= os.path.join(__location__, 'model_ft.pth')
# Define the transforms
def transform_():
    transform = A.Compose(
    [A.Resize(128, 171, always_apply=True),A.CenterCrop(112, 112, always_apply=True),
     A.Normalize(mean = [0.43216, 0.394666, 0.37645],std = [0.22803, 0.22145, 0.216989], always_apply=True)]
     )
    return transform

# To get the mean and std of Custom Dataset
def Mean_and_Std(dataset):
  loader =torch.utils.data.DataLoader(dataset,batch_size=4,num_workers=0,shuffle=False)
  mean = 0.
  std = 0.
  for images, _ in loader:
      batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
      images = images.view(batch_samples, images.size(1), -1) # to reshape it
      mean += images.mean(2).sum(0)
      std += images.std(2).sum(0)

  mean /= len(loader.dataset)
  std /= len(loader.dataset)
  return mean/255,std/255

def frames_extraction(video_path,SEQUENCE_LENGTH):
    '''
    This function will extract the required frames from a video after resizing and normalizing them.
    Args:
        video_path: The path of the video in the disk, whose frames are to be extracted.
        SEQUENCE_LENGTH: TThe number of Frames we want.
    Returns:
        frames_list: A list containing the resized and normalized frames of the video.
    '''

    # Declare a list to store video frames.
    frames_list = []
    
    # Read the Video File using the VideoCapture object.
    video_reader = cv2.VideoCapture(video_path)

    # Get the total number of frames in the video.
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the the interval after which frames will be added to the list.
    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH), 1)

    transform= transform_()

    # Iterate through the Video Frames.
    for frame_counter in range(SEQUENCE_LENGTH):

        # Set the current frame position of the video.
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

        # Reading the frame from the video. 
        success, frame = video_reader.read() 

        # Check if Video frame is not successfully read then break the loop
        if not success:
            break

        image = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transform(image=frame)['image']
        
        # Append the normalized frame into the frames list
        frames_list.append(frame)
    
    # Release the VideoCapture object. 
    video_reader.release()

    # Return the frames list.
    return frames_list


def create_dataset(DATASET_DIR,SEQUENCE_LENGTH):
    '''
    This function will extract the data of the selected classes and create the required dataset.
    Returns:
        features:          A list containing the extracted frames of the videos.
        labels:            A list containing the indexes of the classes associated with the videos.
        video_files_paths: A list containing the paths of the videos in the disk.
    '''

    # Declared Empty Lists to store the features, labels and video file path values.
    features = []
    labels = []
    transform= transform_()
    # Iterating through all the classes mentioned in the classes list
    for class_index, class_name in enumerate(CLASSES_LIST):
        
        # Display the name of the class whose data is being extracted.
        print(f'Extracting Data of Class: {class_name}')
        
        # Get the list of video files present in the specific class name directory.
        files_list=[name for name in glob.glob(DATASET_DIR+'/'+class_name+'/*')]
        
        # Iterate through all the files present in the files list.
        for file_name in files_list:
            framesPath = os.listdir(os.path.join(DATASET_DIR, class_name, file_name))
            frames= [os.path.join(DATASET_DIR, class_name, file_name,Fpath) for Fpath in framesPath]

            frames_list=[]
            for frame in frames:
              frame = cv2.imread (frame, cv2.IMREAD_COLOR)
              image = frame.copy()
              frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
              frame = transform(image=frame)['image']
              frames_list.append(frame)
              
            input_frames= np.array(frames_list)

            # transpose to get [3, num_clips, height, width]
            input_frames = np.transpose(input_frames, (3,0, 1, 2))

            # convert the Frames & Labels to tensor
            input_frames = torch.tensor(input_frames, dtype=torch.float32)
            label = torch.tensor(int(class_index))

            # Append the data to their repective lists and Stack them as Tensor.
            features.append(input_frames) # append to list
            
            labels.append(label) # append to list
            
              
    # Return the frames, class index, and video file path.
    return torch.stack(features), torch.stack(labels)

# Function To Train the Model From Pytorch Documentation
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def testEval(model_ft,optimizer_ft,preTrainedModel,test_dataset,criterion):
  since = time.time()
  test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
  model_ft.eval()
  running_loss = 0.0
  running_corrects = 0
  y_test = []
  y_pred = []
  for inputs, labels in test_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)
    optimizer_ft.zero_grad()
    with torch.set_grad_enabled(False):
      outputs = model_ft(inputs)
      loss = criterion(outputs, labels)
      _, preds = torch.max(outputs, 1)
    running_loss += loss.item() * inputs.size(0)
    running_corrects += torch.sum(preds == labels.data)
    y_test += labels.data.tolist()
    y_pred += preds.data.tolist()

  epoch_loss = running_loss / len(test_loader.dataset)
  epoch_acc = running_corrects.double() / len(test_loader.dataset)
  time_elapsed = time.time() - since
  print(preTrainedModel)
  print("--------------")
  print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
  print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
  print("--------------"*2)
  return y_test, y_pred


def confusionMatrixPlot(y_test, y_pred,preTrainedModel):
  cf_matrix= confusion_matrix(y_test, y_pred)
  ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')
  print(preTrainedModel)
  print("--------------")
  ax.set_title('Confusion Matrix\n\n');
  ax.set_xlabel('\nPredicted Values')
  ax.set_ylabel('Actual Values ');
  ## Ticket labels - List must be in alphabetical order
  ax.xaxis.set_ticklabels(['accident','no_accident'])
  ax.yaxis.set_ticklabels(['accident','no_accident'])
  ## Display the visualization of the Confusion Matrix.
  plt.show()

def classificationReport(y_test,y_pred,preTrainedModel):
    target_names = ["accident","no_accident"]
    print(preTrainedModel)
    print("--------------")
    print(classification_report(y_test, y_pred, target_names=target_names))


def loadModel():
  model_ft = torchvision.models.video.r3d_18(pretrained=True, progress=False)
  num_ftrs = model_ft.fc.in_features         #in_features
  model_ft.fc = torch.nn.Linear(num_ftrs, 2) #nn.Linear(in_features, out_features)
  model_ft.load_state_dict(torch.load(modelWeights,map_location=torch.device(device)))
  model_ft.to(device)
  model_ft.eval()
  return model_ft

model = loadModel()

def PredTopKClass(k, clips):
  with torch.no_grad(): # we do not want to backprop any gradients

      input_frames = np.array(clips)
      
      # add an extra dimension        
      input_frames = np.expand_dims(input_frames, axis=0)

      # transpose to get [1, 3, num_clips, height, width]
      input_frames = np.transpose(input_frames, (0, 4, 1, 2, 3))

      # convert the frames to tensor
      input_frames = torch.tensor(input_frames, dtype=torch.float32)
      input_frames = input_frames.to(device)

      # forward pass to get the predictions
      outputs = model(input_frames)

      # get the prediction index
      soft_max = torch.nn.Softmax(dim=1)  
      probs = soft_max(outputs.data) 
      prob, indices = torch.topk(probs, k)

  Top_k = indices[0]
  Classes_nameTop_k=[CLASSES_LIST[item].strip() for item in Top_k]
  ProbTop_k=prob[0].tolist()
  ProbTop_k = [round(elem, 5) for elem in ProbTop_k]
  return Classes_nameTop_k[0] , ProbTop_k[0]  #list(zip(Classes_nameTop_k,ProbTop_k))


def PredTopKProb(k,clips):
  with torch.no_grad(): # we do not want to backprop any gradients

      input_frames = np.array(clips)
      
      # add an extra dimension        
      input_frames = np.expand_dims(input_frames, axis=0)

      # transpose to get [1, 3, num_clips, height, width]
      input_frames = np.transpose(input_frames, (0, 4, 1, 2, 3))

      # convert the frames to tensor
      input_frames = torch.tensor(input_frames, dtype=torch.float32)
      input_frames = input_frames.to(device)

      # forward pass to get the predictions
      outputs = model(input_frames)

      # get the prediction index
      soft_max = torch.nn.Softmax(dim=1)  
      probs = soft_max(outputs.data) 
      prob, indices = torch.topk(probs, k)

  Top_k = indices[0]
  Classes_nameTop_k=[CLASSES_LIST[item].strip() for item in Top_k]
  ProbTop_k=prob[0].tolist()
  ProbTop_k = [round(elem, 5) for elem in ProbTop_k]
  return list(zip(Classes_nameTop_k,ProbTop_k))


def downloadYouTube(videourl, path):

    yt = YouTube(videourl)
    yt = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
    if not os.path.exists(path):
        os.makedirs(path)
    yt.download(path)

def show_video(file_name, width=640):
  # show resulting deepsort video
  mp4 = open(file_name,'rb').read()
  data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
  return HTML("""
  <video width="{0}" controls>
        <source src="{1}" type="video/mp4">
  </video>
  """.format(width, data_url))

def accidentInference(video_path,SEQUENCE_LENGTH=64):
  clips = frames_extraction(video_path,SEQUENCE_LENGTH)
  print(PredTopKClass(1,clips))
  print(PredTopKProb(2,clips))
  return "***********"


def accidentInference_Time(video_path,SEQUENCE_LENGTH=64):
  start_time = time.time()
  clips = frames_extraction(video_path,SEQUENCE_LENGTH)
  class_=PredTopKClass(1,clips)
  elapsed = time.time() - start_time
  print("time is:",elapsed)
  return class_




def predict_on_video(video_file_path, output_file_path,SEQUENCE_LENGTH,skip=2,showInfo=False,thresholding=0.85):
    '''
    This function will perform action recognition on a video using the LRCN model.
    Args:
    video_file_path:  The path of the video stored in the disk on which the action recognition is to be performed.
    output_file_path: The path where the ouput video with the predicted action being performed overlayed will be stored.
    SEQUENCE_LENGTH:  The fixed number of frames of a video that can be passed to the model as one sequence.
    '''

    # Initialize the VideoCapture object to read from the video file.
    video_reader = cv2.VideoCapture(video_file_path)

    # Get the width and height of the video.
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize the VideoWriter Object to store the output video in the disk.
    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 
                                   video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))

    # Declare a queue to store video frames.
    frames_queue = deque(maxlen = SEQUENCE_LENGTH)
    transform= transform_()
    # Initialize a variable to store the predicted action being performed in the video.
    predicted_class_name = ''

    # Iterate until the video is accessed successfully.
    counter=0
    while video_reader.isOpened():

        # Read the frame.
        ok, frame = video_reader.read() 
        
        # Check if frame is not read properly then break the loop.
        if not ok:
            break

        image = frame.copy()
        framee = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        framee = transform(image=framee)['image']
        if counter % skip==0:
          # Appending the pre-processed frame into the frames list.
          frames_queue.append(framee)
         
        
        # Check if the number of frames in the queue are equal to the fixed sequence length.
        if len(frames_queue) == SEQUENCE_LENGTH:
          predicted_class_name,prob= PredTopKClass(1,frames_queue)
          if showInfo:
            print(predicted_class_name,prob)
            frames_queue = deque(maxlen = SEQUENCE_LENGTH)
          else:
            frames_queue = deque(maxlen = SEQUENCE_LENGTH)
    
        # Write predicted class name on top of the frame.
        if (predicted_class_name=="accident") and (prob >= thresholding) :
        	textsize = cv2.getTextSize("accident", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        	# print(textsize[0])
        	# print(textsize[1])
        	# print(frame.shape[1])
        	# print(frame.shape[0])
        	textX = int((frame.shape[1] - textsize[0]) / 2)
        	textY = int((frame.shape[0] + textsize[1]) / 2)

        	cv2.putText(frame, predicted_class_name, (textX, textY), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
        	textsize = cv2.getTextSize("no_accident", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        	# print(textsize[0])
        	# print(textsize[1])
        	# print(frame.shape[1])
        	# print(frame.shape[0])
        	textX = int((frame.shape[1] - textsize[0]) / 2)
        	textY = int((frame.shape[0] + textsize[1]) / 2)
        	
        	cv2.putText(frame, "no_accident", (textX, textY), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        counter+=1
        
        # Write The frame into the disk using the VideoWriter Object.

        video_writer.write(frame)
        # time.sleep(2)
    if showInfo:
      print(counter)  
    # Release the VideoCapture and VideoWriter objects.
    video_reader.release()
    video_writer.release()

def showIference(sequence,skip,input_video_file_path,output_video_file_path,showInfo=False,thresholding=0.85):
    # Perform Accident Detection on the Test Video.
    predict_on_video(input_video_file_path, output_video_file_path,sequence,skip,showInfo,thresholding)
    return output_video_file_path

def crashDetection(inputPath,seq,skip,outputPath,showInfo=False,thresholding=0.85):
    # Perform Accident Detection on the Test Video.
    predict_on_video(inputPath, outputPath,seq,skip,showInfo,thresholding)
    return outputPath

def streaming_framesInference(frames,thresholding=0.85):
    clips = []
    transform = transform_()
    for frame in frames:
        image = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transform(image=frame)['image']

        # Append the normalized frame into the frames list
        clips.append(frame)
    class_c,probb = PredTopKClass(1, clips)
    print(class_c,probb)
    if (class_c=="accident") and (probb >= thresholding) :
      return class_c
    else:
      return "no_accident"

def streaming_predict(frames,thresholding=0.85):
    prediction= streaming_framesInference(frames,thresholding)
    global predicted_class_name
    predicted_class_name = prediction


def start_streaming(streamingUrl,thresholding=0.85):
    video = cv2.VideoCapture(streamingUrl)
    l = []
    last_time = time.time() - 3
    while True:
        _, frame = video.read()
        if last_time+2.5 < time.time():
            l.append(frame)
        if len(l) == 16:
            last_time = time.time()
            x = threading.Thread(target=streaming_predict, args=(l,thresholding))
            x.start()
            l = []
        if predicted_class_name == "accident":
        	
        	textsize = cv2.getTextSize("accident", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        	# print(textsize[0])
        	# print(textsize[1])
        	# print(frame.shape[1])
        	# print(frame.shape[0])
        	textX = int((frame.shape[1] - textsize[0]) / 2)
        	textY = int((frame.shape[0] + textsize[1]) / 2)
        	cv2.putText(frame, predicted_class_name, (textX, textY), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
        	
        	textsize = cv2.getTextSize("no_accident", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        	# print(textsize[0])
        	# print(textsize[1])
        	# print(frame.shape[1])
        	# print(frame.shape[0])
        	textX = int((frame.shape[1] - textsize[0]) / 2)
        	textY = int((frame.shape[0] + textsize[1]) / 2)
        	cv2.putText(frame, "no_accident", (textX, textY), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("RTSP", frame)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()