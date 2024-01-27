import cv2
import mediapipe as mp
import time 


class handDetector():
    def __init__(self, mode=False,maxHands=2, model_complexity=1, detectionConf=0.5, trackConf=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.model_complexity=model_complexity
        self.detectionConf = detectionConf
        self.trackConf = trackConf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.model_complexity, self.detectionConf,self.trackConf) 
        self.mpDraw = mp.solutions.drawing_utils

        self.fingerTipsId = [4,8,12,16,20]
    
    def findHands(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        # print(results)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
    
    def findPosition(self,img, handNo=0, draw=True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id,lm)
                h,w,c = img.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                self.lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),5,(255,0,255), cv2.FILLED)
        return self.lmList
    
    def FingersUp(self): 
        fingers=[]
        
        if self.lmList[self.fingerTipsId[0]][1] < self.lmList[self.fingerTipsId[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # fingers     
        for id in range(1,5):
            if self.lmList[self.fingerTipsId[id]][2] < self.lmList[self.fingerTipsId[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers


                


def main():
    pTime = 0 
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        myList = detector.findPosition(img)
        if len(myList) != 0:
            print(myList[4])
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime=cTime

        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_COMPLEX,3,(255,0,255),3)
        cv2.imshow("image",img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
