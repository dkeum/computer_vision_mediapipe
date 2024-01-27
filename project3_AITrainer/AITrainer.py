import cv2
import time
import mediapipe as mp
import numpy as np

import sys
sys.path.append(".") 
from Pose_Estimation_basic import pose_estimation as pe

wCam, hCam = 640,480

cap= cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)


pTime =0

# store dumbbell curl count
count = 0
dir = 0

detector = pe.poseDetector()



while True:
    sucess, img = cap.read()

    img = detector.findPose(img,draw=False)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) !=0:

        #right arm 
        # img, angle = detector.findAngle(img, 12,14,16)

        #left arm
        img, angle = detector.findAngle(img, 11,13,15)

        armBendPerc = np.interp(angle, (185,345), (0,100))
        

        # print(angle, armBendPerc)

        if armBendPerc== 100:
            if dir==0:
                count +=0.5
                dir=1
        if armBendPerc==0:
            if dir==1:
                count+=0.5
                dir=0

        # arm bend bar
        xbar, xbar_width  = 25, 40
        ybar, ybar_height = 100, 120
        bar= np.interp(angle, (185,345),(ybar+ybar_height,ybar))

        cv2.rectangle(img, (xbar,ybar),(xbar+xbar_width,ybar+ybar_height), (0,255,0),2)
        if armBendPerc <= 50: 
            cv2.rectangle(img,(xbar,int(bar)),(xbar+xbar_width,ybar+ybar_height), (0,255,0), cv2.FILLED)
        elif armBendPerc > 50 and armBendPerc <80: 
            cv2.rectangle(img,(xbar,int(bar)),(xbar+xbar_width,ybar+ybar_height), (0,255,255), cv2.FILLED)
        else: 
            cv2.rectangle(img,(xbar,int(bar)),(xbar+xbar_width,ybar+ybar_height), (0,0,255), cv2.FILLED)

        cv2.putText(img, f'{int(armBendPerc)}%',(xbar,ybar+ybar_height+40), cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

        # count the num of curls
        cv2.rectangle(img,(0,350),(150,480), (0,255,0),cv2.FILLED)
        cv2.putText(img, f'{int(count)}',(10,450), cv2.FONT_HERSHEY_PLAIN,7,(255,0,0),7)




        

            

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime=cTime

    cv2.putText(img,f'FPS: {int(fps)}', (400,70), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
    cv2.imshow("img", img)
    cv2.waitKey(1)