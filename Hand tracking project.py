# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 15:49:19 2021

@author: Souli
"""

import cv2 
import mediapipe as mp 
import time 

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0 
cTime = 0
while True: 
    success, img = cap.read()
    imgRBG = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRBG)
    print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for ID, lm in enumerate(handLms.landmark):
                h,w,c = img.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                print(ID,cx,cy)
                if ID == 8:
                    cv2.circle(img, (cx,cy),15,(255,0,255),cv2.FILLED)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    
    
    cv2.imshow('cam',img)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()


#