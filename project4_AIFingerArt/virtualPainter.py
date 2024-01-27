import cv2
import numpy as np
import os
import time
import sys

sys.path.append(".") 
from HandTracking_basic import Hand_Tracking_module as htm

folderPath = "project4_AIFingerArt/paintBrushMenu"
myList = os.listdir(folderPath)


overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    # print(f'{folderPath}/{imPath}')
    overlayList.append(image)

header = overlayList[0]
drawColor = (0,0,255)
brushThickness =5
eraserThickness = 100
xp,yp =0,0


cap= cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

pTime =0

detector = htm.handDetector(detectionConf=0.85)

imgCanvas = np.zeros((720,1280,3),np.uint8)

while True:
    sucess, img = cap.read()
    img = cv2.resize(img, (1280,720))
    img = cv2.flip(img,1)
   
    img = detector.findHands(img)
    lmList = detector.findPosition(img,draw=False)

    

    if len(lmList)!= 0:
        fingersUp = detector.FingersUp()
        #tip of index and middle finger
        x1,y1 = lmList[8][1:]
        x2,y2 = lmList[12][1:]

        if fingersUp[1] and fingersUp[2]:
            xp,yp = 0 , 0
            print("selection Mode")

            if y1 < 136:
                if  300<x1<550:
                    header=overlayList[0]
                    drawColor=(0,0,255)
                elif 551<x1<780:
                    header=overlayList[1]
                    drawColor=(0,255,0)
                elif 781<x1<1020:
                    header=overlayList[2]
                    drawColor=(255,0,0)
                elif 1021<x1<1280:
                    header=overlayList[3]
                    drawColor=(0,0,0)
            cv2.rectangle(img, (x1,y1-25), (x2,y2+25),drawColor,cv2.FILLED)
                    
        elif fingersUp[1] and fingersUp[2] != 1:
            cv2.circle(img, (x1,y1), 15 ,drawColor, cv2.FILLED)
            print("drawing mode")
            
            if xp == 0 and yp ==0:
                xp,yp=x1,y1
            
            if drawColor==(0,0,0):
                cv2.line(img, (xp,yp), (x1,y1),drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp,yp), (x1,y1),drawColor, eraserThickness)

            cv2.line(img, (xp,yp), (x1,y1),drawColor, brushThickness)
            cv2.line(imgCanvas, (xp,yp), (x1,y1),drawColor, brushThickness)
            xp,yp = x1,y1

    imgGray = cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50,255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)

    h,w,c = header.shape
    img[0:h, 0:w] = header
    # img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime=cTime

    cv2.putText(img,f'FPS: {int(fps)}', (400,70), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
    cv2.imshow("img", img)
    # cv2.imshow("imgcanvas", imgCanvas)
    cv2.waitKey(1)