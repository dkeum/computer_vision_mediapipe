import cv2
import mediapipe as mp
import time 

class FaceMesh():
    def __init__(self, staticMode=False,maxFaces=2,minDectConf=0.5, minTrackConf=0.5):
        self.staticMode=staticMode
        self.maxFaces=maxFaces
        self.minDectConf=minDectConf
        self.minTrackConf=minTrackConf
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(max_num_faces=self.maxFaces)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces=[]
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(img,faceLms,self.mpFaceMesh.FACEMESH_TESSELATION, landmark_drawing_spec=self.drawSpec)

                face=[]
                for id,lm in enumerate(faceLms.landmark):
                    ih,iw,ic = img.shape
                    x,y = int(lm.x*iw), int(lm.y*ih)
                    cv2.putText(img, str(id), (x,y), cv2.FONT_HERSHEY_COMPLEX,0.3,(0,255,0),1)
                    face.append([x,y])
                faces.append(face)

        return img,faces 
    
    


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0

    detector = FaceMesh()
    while True:
        success, img = cap.read()

        img ,faces = detector.findFaceMesh(img)
        if len(faces) !=0:
            print(len(faces))

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img,f'FPS: {int(fps)}',(20,70),cv2.FONT_HERSHEY_COMPLEX,3,(0,255,0),2)
        cv2.imshow("image",img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()