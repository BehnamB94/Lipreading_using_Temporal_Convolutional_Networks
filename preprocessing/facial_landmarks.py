import cv2
import os
import urllib.request as urlreq
import numpy as np
from tempfile import TemporaryFile


# save face detection algorithm's url in haarcascade_url variable
haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"

# save face detection algorithm's name as haarcascade
haarcascade = "haarcascade_frontalface_alt2.xml"
haarcascade_clf = "data/" + haarcascade

# check if data folder is in working directory
if (os.path.isdir('data')):
    # check if haarcascade is in data directory
    if (haarcascade in os.listdir('data')):
        print("File exists")
    else:
        # download file from url and save locally as haarcascade_frontalface_alt2.xml
        urlreq.urlretrieve(haarcascade_url, haarcascade_clf)
        print("File downloaded")
else:
    # create data folder in current directory
    os.mkdir('data')
    # download haarcascade to data folder
    urlreq.urlretrieve(haarcascade_url, haarcascade_clf)
    print("File downloaded")

# create an instance of the Face Detection Cascade Classifier
detector = cv2.CascadeClassifier(haarcascade_clf)

# save facial landmark detection model's url in LBFmodel_url variable
LBFmodel_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"

# save facial landmark detection model's name as LBFmodel
LBFmodel = "LFBmodel.yaml"
LBFmodel_file = "data/" + LBFmodel

# check if data folder is in working directory
if (os.path.isdir('data')):
    # check if Landmark detection model is in data directory
    if (LBFmodel in os.listdir('data')):
        print("File exists")
    else:
        # download file from url and save locally as haarcascade_frontalface_alt2.xml
        urlreq.urlretrieve(LBFmodel_url, LBFmodel_file)
        print("File downloaded")
else:
    # create data folder in current directory
    os.mkdir('data')
    # download Landmark detection model to data folder
    urlreq.urlretrieve(LBFmodel_url, LBFmodel_file)
    print("File downloaded")

# create an instance of the Facial landmark Detector with the model
landmark_detector  = cv2.face.createFacemarkLBF()
landmark_detector.loadModel(LBFmodel_file)

def get_landmarks(video_path, npz_out_path):
    cap = cv2.VideoCapture(video_path)
    temp = []
    if (cap.isOpened() == False):
        print("error")
    i = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if(ret == True):
            i += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray)
            for (x,y,w,d) in faces:
                #print(faces)
                _, landmarks = landmark_detector.fit(gray, np.array(faces))
                temp.append(landmarks[0])
                break
        if ret == False:
            break
    cap.release()
    np.savez_compressed(npz_out_path, data=temp)
