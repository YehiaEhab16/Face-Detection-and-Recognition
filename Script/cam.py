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
import torch
import torchvision
from torch import nn
import cv2

# Trained Model Path
CURRENCY_MODEL_PATH = 'currency_model.pth'
FACE_MODEL_PATH     = 'face_model.pth'

# Camera Thread
class Camera(QThread):
    imageUpdate = pyqtSignal(np.ndarray)
    def __init__(self,mainWindow):
        super(Camera, self).__init__()
        self.mainWindow = mainWindow
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

        # PyTorch Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.class_currency = ['egp_10', 'egp_100', 'egp_10_new', 'egp_20', 'egp_200', 'egp_20_new', 'egp_5', 'egp_50']
        self.class_faces = ['Mohamed','Yehia']

        # Load Trained Model
        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        self.transform = weights.transforms()
        self.currency_model = torchvision.models.efficientnet_b0(weights=weights).to(self.device)
        self.face_model = torchvision.models.efficientnet_b0(weights=weights).to(self.device)

        # Adjust model
        for param in self.currency_model.features.parameters():
            param.requires_grad = False

        for param in self.face_model.features.parameters():
            param.requires_grad = False

        self.currency_model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=len(self.class_currency))
            ).to(self.device)
        
        self.face_model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=len(self.class_faces))
            ).to(self.device)

        # Load previously trained weights
        self.currency_model.load_state_dict(torch.load(f=CURRENCY_MODEL_PATH, map_location=torch.device(self.device)))
        self.face_model.load_state_dict(torch.load(f=FACE_MODEL_PATH, map_location=torch.device(self.device)))


    # Kill Thread
    def terminate(self):
        self.ThreadActive=False
        self.Capture.release()      # Release Camera

    def setHandsMode(self):
        self.mode='hands'

    def setFaceMode(self):
        self.mode='face'
    
    def setCurrencyMode(self):
        self.mode='currency'

    def rotateRight(self):
        if self.orientation==2:
            self.orientation=-1
        else:
            self.orientation+=1
    
    def rotateLeft(self):
        if self.orientation==-2:
            self.orientation=1
        else:
            self.orientation-=1

    def hand_mode(self,frame):
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
        else:
            self.mainWindow.detected.setText('unkown')
            self.mainWindow.number.setText('xx')

        # Display finger count
        try:
            self.mainWindow.detected.setText(handLabel)
            self.mainWindow.number.setText(str(fingerCount))
        except UnboundLocalError:
            pass

    def face_mode(self,frame):
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

            self.predict_image_class(frame,self.face_model)
        
    # Predict image class
    def currency_mode(self,frame):    
        self.predict_image_class(frame,self.currency_model)

    def predict_image_class(self,frame,model):
        # 1. Load in image and convert the tensor values to float32
        target_image =  torch.from_numpy(frame).permute(2, 0, 1).type(torch.float32)
        # 2. Divide the image pixel values by 255 to get them between [0, 1]
        target_image = target_image / 255. 
        # 3. Transform if necessary
        target_image = self.transform(target_image)
        # 4. Make sure the model is on the target device
        model.to(self.device)
        # 5. Turn on model evaluation mode and inference mode
        model.eval()

        # Turn on inference context manager
        with torch.inference_mode():
            # Add an extra dimension to the image
            target_image = target_image.unsqueeze(dim=0)
            # Make a prediction on image with an extra dimension and send it to the target device
            target_image_pred = model(target_image.to(self.device))
            
        # 6. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
        target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
        # 7. Convert prediction probabilities -> prediction labels
        target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
        
        # 8. Plot the image alongside the prediction and prediction probability
        if target_image_pred_probs.max().cpu() > 0.7:
            if model == self.currency_model:
                self.mainWindow.detected.setText(self.class_currency[target_image_pred_label.cpu()])
            else:
                self.mainWindow.detected.setText(self.class_faces[target_image_pred_label.cpu()])
            self.mainWindow.number.setText(f'{target_image_pred_probs.max().cpu()*100:.2f}%')
        else:
            self.mainWindow.detected.setText('unkown')
            self.mainWindow.number.setText('xx.xx%')
            
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
                self.hand_mode(frame)

            elif self.mode=='face':
                self.face_mode(frame)
            
            elif self.mode=='currency':
                self.currency_mode(frame)

            # Display RGB Image
            Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                

            # Update Image
            self.imageUpdate.emit(Image)