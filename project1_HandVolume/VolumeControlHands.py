import cv2
import mediapipe as mp
import time
import numpy as np
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

import sys
sys.path.append(".") 
# run python .\project1_HandVolume\VolumeControlHands.py
# from the path: (.venv) PS C:\Users\16047\Desktop\python_project\computer_vision> 
from HandTracking_basic import Hand_Tracking_module as htm


#############################
wCam, hCam = 640,480
#############################

cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)


######### PYCAW FOR AUDIO/VOLUME CONTROL#############

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange() # (-96.0, 0.0, 0.125)
minVol= volRange[0]
maxVol= volRange[1]
vol = 0
volBar= 0 
volPercentage=0

##########


pTime=0
detector = htm.handDetector()
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img,draw=False)
    if len(lmList) != 0:
        x1,y1 = lmList[4][1], lmList[4][2]
        x2,y2 = lmList[8][1], lmList[8][2]
        cx,cy = (x1+x2)//2, (y1+y2)//2

        cv2.circle(img, (x1,y1), 15,(255,0,255),cv2.FILLED)
        cv2.circle(img, (x2,y2), 15,(255,0,255),cv2.FILLED)
        cv2.circle(img, (cx,cy), 15,(255,0,255),cv2.FILLED)
        cv2.line(img, (x1,y1), (x2,y2),(255,0,255),3)

        length = math.hypot(x2-x1,y2-y1)

        # Map hand range (50,300) to audio range (-96,0)
        vol = np.interp(length, [30,330],[minVol,maxVol])
        volBar = np.interp(length, [30,330],[400,150])
        volPercentage = np.interp(length, [30,330],[0,100])
        volume.SetMasterVolumeLevel(vol, None)

        if length < 50:
            cv2.circle(img, (cx,cy), 15,(0,255,0),cv2.FILLED)
        
    cv2.rectangle(img, (50,150),(85,400),(0,255,0),3) # volBar container
    cv2.rectangle(img, (50,int(volBar)),(85,400),(255,0,0),cv2.FILLED) # volBar

    cv2.putText(img,f'{int(volPercentage)}%', (40,450), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1) # vol %
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime=cTime

    cv2.putText(img,f'FPS: {int(fps)}', (40,50), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
    cv2.imshow("img", img)
    cv2.waitKey(1)