##########################################################################################
####################### Project: Face Detection and Recognition    #######################
####################### Version: 1.1                               #######################
####################### Authors: Yehia Ehab                        #######################
####################### Date : 10/06/2024                          #######################
##########################################################################################

# Camera class #

# Importing PyQT packages
from PyQt5.QtCore import QThread,pyqtSignal

# Importing Image Modules
import mediapipe as mp
import numpy as np
import cv2

# Camera Thread
class Camera(QThread):
    imageUpdate = pyqtSignal(np.ndarray)
    def __init__(self,parent=None):
        super(Camera, self).__init__(parent)
        self.orientation=0
        self.mode=None

        # Fingers dictionary
        self.fingers = dict()

        # Media Pipe Hand detection
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(max_num_hands=2)
        self.mpDraw = mp.solutions.drawing_utils

        # Media Pipe Face Detection
        mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = mpFaceDetection.FaceDetection()

        self.url= 0


    # Kill Thread
    def terminate(self):
        self.ThreadActive=False
        self.Capture.release()      # Release Camera

    def setHandsMode(self):
        self.mode='hands'

    def setFaceMode(self):
        self.mode='face'

    def rotate(self,direction):
        # Rotate Right
        if direction == 'right':
            if self.orientation==2:
                self.orientation=-1
            else:
                self.orientation+=1
		# Rotate Left
        elif direction == 'left':
            if self.orientation==-2:
                self.orientation=1
            else:
                self.orientation-=1

    def run(self):
        try:
            self.Capture = cv2.VideoCapture(self.url, cv2.CAP_DSHOW) # Camera URL
            self.ThreadActive = True
        except:
            self.ThreadActive = False
        while self.ThreadActive:
            ret, frame = self.Capture.read()    # Read Camera
            if ret == False:
                continue
                
            frame = cv2.flip(frame,1)

            # Rotate Right  
            if self.orientation==1:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
		    # Rotate Left
            elif self.orientation==-1:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
		    # Flip
            elif self.orientation==-2 or self.orientation==2:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            
            # Face Detection and Recognition
            if self.mode== 'hands':
                fingerCount=0
                # Convert to RGB
                imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Detect Hands
                results = self.hands.process(imgRGB)

                # Show Detected Hands
                if results.multi_hand_landmarks:
                    for handLms in results.multi_hand_landmarks:
                        # Get hand index to check label (left or right)
                        handIndex = results.multi_hand_landmarks.index(handLms)
                        handLabel = results.multi_handedness[handIndex].classification[0].label

                        # Set variable to keep landmarks positions (x and y)
                        handLandmarks = []
                        self.mpDraw.draw_landmarks(frame,handLms,self.mpHands.HAND_CONNECTIONS)
                        for id,lm in enumerate(handLms.landmark):
                            handLandmarks.append([lm.x, lm.y])
                            h,w,_ = frame.shape
                            cx,cy=int(lm.x * w), int(lm.y * h)
                            self.fingers[id]=[cx,cy]
                        if handLabel == "Left":
                            if handLandmarks[4][0] > handLandmarks[3][0]:
                                fingerCount = fingerCount+1
                                
                        elif handLabel == "Right":
                            if handLandmarks[4][0] < handLandmarks[3][0]:
                                fingerCount = fingerCount+1

                        if handLandmarks[8][1] < handLandmarks[6][1]:       #Index finger
                            fingerCount = fingerCount+1

                        if handLandmarks[12][1] < handLandmarks[10][1]:     #Middle finger
                            fingerCount = fingerCount+1

                        if handLandmarks[16][1] < handLandmarks[14][1]:     #Ring finger
                            fingerCount = fingerCount+1

                        if handLandmarks[20][1] < handLandmarks[18][1]:     #Pinky
                            fingerCount = fingerCount+1

                # Display finger count
                try:
                    cv2.putText(frame, f'{handLabel} Hand - {fingerCount} fingers', (20, 450), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 5)
                except UnboundLocalError:
                    pass

            elif self.mode=='face':
                 # Convert to RGB
                imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Detect the faces
                results = self.faceDetection.process(imgRGB)

                # Show Detected Faces
                if results.detections:
                    for detection in results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        ih,iw,_ = frame.shape
                        x,y,w,h = int(bboxC.xmin*iw),int(bboxC.ymin*ih),int(bboxC.width*iw),int(bboxC.height*ih)

                        # Draw Boundary Box around face
                        cv2.rectangle(frame,[x,y,w,h],(255,0,0),2)
                        cv2.putText(frame, 'Yehia', (x+5,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # Display RGB Image
            Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                

            # Update Image
            self.imageUpdate.emit(Image)