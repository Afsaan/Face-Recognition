from facenet_pytorch import MTCNN
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
from PIL import Image
from tabulate import tabulate
import datetime
import pickle
import argparse

import os

ap = argparse.ArgumentParser()
ap.add_argument("-v" , "--videos",  help = "path to the video" , default = 0)
args = vars(ap.parse_args())

model = load_model('models/facenet_keras.h5')
# svm_model = joblib.load('models/svm_face_classification.pkl')
with open('models/svm_classification.pkl', 'rb') as file:
    svm_model = pickle.load(file)

mtcnn = MTCNN( keep_all = True , post_process = False)

names_array = ['Afsan' , 'Amresh' , 'Amritansh' , 'Ayush' , 'Harish' , 'Keyur' , 'Rahul']

video = cv2.VideoCapture(args['videos'])
cv2.namedWindow('face Recognition', cv2.WINDOW_NORMAL)
cv2.resizeWindow('face Recognition', 800, 800)
loop = True
while loop:
    ret , frame = video.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    try:
        faces = mtcnn(frame_rgb)
        boxes, _ = mtcnn.detect(frame_rgb)
        print(f'number of faces {len(faces)}')
        for i in range(len(faces)):
            face = faces[i].permute(1, 2, 0).int().numpy().reshape(1,160,160,3) / 255.0
            yhat = model.predict(face)
            print(yhat.shape)
            name_class = svm_model.predict(yhat[0].reshape(1,128))
            print(names_array[name_class[0]])
            frame = cv2.rectangle(frame , (boxes[i][0] , boxes[i][1]) , (boxes[i][2],boxes[i][3]) , (0, 255, 0) , 2)

            frame = cv2.rectangle(frame, (int(boxes[i][0]), int(boxes[i][1]-5)), (int(boxes[i][0]) + 120, int(boxes[i][1]-10) -30), (255,255,255), -1)
            frame = cv2.putText(frame, names_array[name_class[0]] , (int(boxes[i][0]) , int(boxes[i][1]-10))  , cv2.FONT_HERSHEY_SIMPLEX  , 1 , (0, 0, 0) , 2)
            cv2.imshow('face Recognition' , frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    except :
        continue