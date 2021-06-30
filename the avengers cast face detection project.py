# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 18:12:57 2021

@author: Souli
"""

import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

 
path = 'F:/face detection project/the avengers/the5'
images = []
Names = []
myList = os.listdir(path)
print(myList)
for justName in myList:
    curImg = cv2.imread(f'{path}/{justName}')
    images.append(curImg)
    Names.append(os.path.splitext(justName)[0])
print(Names)


def resize_func(img,scale):
    #resize function 
    if img.shape[0] > scale:
        # To declare how much to resize
        resize_scaling = int(scale*100 / img.shape[0]) 
        resize_width = int(img.shape[1] * resize_scaling/100)
        resize_hieght = int(img.shape[0] * resize_scaling/100) 
        resized_dimentions = (resize_width, resize_hieght) 
        # Create resized image using the calculated dimentions 
        resized_img = cv2.resize(img, resized_dimentions,interpolation=cv2.INTER_AREA)
    return resized_img,resize_scaling

def Encodings(photo):
    encodeList = []
    for img in photo:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
 
encodeListKnown = Encodings(images)
print('Encoding Finished')
 
cap = cv2.VideoCapture(0)
 
while True:
    success, img = cap.read()
    imgS,scale = resize_func(img,75)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
    facesCurrentFrame = face_recognition.face_locations(imgS)
    encodesCurrentFrame = face_recognition.face_encodings(imgS,facesCurrentFrame)
    
    for encodeFace,faceLoc in zip(encodesCurrentFrame,facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        matchIndex = np.argmin(faceDis)
        
        if faceDis[matchIndex]< 0.60:
            name = Names[matchIndex].upper()
            boundingBoxColor = (0,255,0)
        else: 
            name = 'Unknown'
            boundingBoxColor = (0,0,255)
        Top,Right,Bottom,Left = faceLoc
        Top, Right, Bottom, Left = int(Top*(100/scale)),int(Right*(100/scale)),int(Bottom*(100/scale)),int(Left*(100/scale))
        cv2.rectangle(img,(Left,Top),(Right,Bottom),boundingBoxColor,2)
        cv2.rectangle(img,(Left,Bottom-35),(Right,Bottom),boundingBoxColor,cv2.FILLED)
        cv2.putText(img,name,(Left+6,Bottom-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
    cv2.imshow('Webcam',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()