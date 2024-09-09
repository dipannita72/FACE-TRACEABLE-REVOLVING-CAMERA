# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 20:49:04 2020

@author: hp
"""

import cv2
import os
import shutil
import dlib
import scipy.misc
import numpy as np
import face_recognition
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mtcnn.mtcnn import MTCNN


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
detector = MTCNN()

cap = cv2.VideoCapture(0)
i=0
while True: 
    face_box_list = [] # info (x,y,w,h) for all the faces in the current frame will be stored in this list
    cnt = 1
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    # Detect the faces
    faces =  detector.detect_faces(img)# detectMultiScale(img_type, scale_factor, min_neighbor)
    
    i+=1
    timer = cv2.getTickCount()
    for result in faces:
        #face_box_list.append((x, y, w, h)) # face info will be appended in the list
        #cv2.putText(img, 'ID '+str(cnt), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,255,100), 2) # put the id no with the bounding boxes
        x, y, w, h = result['box']
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2) # put bounding box on the faces
        box = []
        box.append((y, x+w, y+h, x))
        crop_img = img[y:y+h, x:x+w]
        cv2.imshow('x',crop_img)
        path = r'F:\Zaima\face_recognition\images'
        directory = r'F:\Zaima\face_recognition'
        os.chdir(path) 
        cv2.imwrite(str(i)+'.jpg',crop_img)
        os.chdir(directory) 
        
   
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
    
   
        
        
cap.release()
