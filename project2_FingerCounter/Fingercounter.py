import cv2
import time
import mediapipe as mp
import os
import sys
sys.path.append(".") 
from HandTracking_basic import Hand_Tracking_module as htm

wCam, hCam = 640,480

cap= cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

folderPath='project2_FingerCounter/FingerImages'
myList = os.listdir(folderPath)

overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    # print(f'{folderPath}/{imPath}')
    overlayList.append(image)

pTime =0

detector = htm.handDetector(detectionConf=0.75)

fingerTipsId = [4,8,12,16,20]

while True:
    sucess, img = cap.read()

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) !=0:
        fingers=[]

        # thumb
        if lmList[fingerTipsId[0]][1] < lmList[fingerTipsId[0]-1][1]:
                fingers.append(1)
        else:
             fingers.append(0)
        # fingers     
        for id in range(1,5):
            if lmList[fingerTipsId[id]][2] < lmList[fingerTipsId[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)
        totalfingers= fingers.count(1)    
        h,w,c = overlayList[totalfingers].shape
        img[0:h,0:w] = overlayList[totalfingers]
        cv2.rectangle(img, (20,225),(170,425),(0,255,0), cv2.FILLED)
        cv2.putText(img,f'{totalfingers}',(45,375),cv2.FONT_HERSHEY_PLAIN,10,(255,0,0),25)
        

            

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime=cTime

    cv2.putText(img,f'FPS: {int(fps)}', (400,70), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
    cv2.imshow("img", img)
    cv2.waitKey(1)