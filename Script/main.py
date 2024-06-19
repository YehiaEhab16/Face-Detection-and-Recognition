##########################################################################################
####################### Project: Face Detection and Recognition    #######################
####################### Version: 1.1                               #######################
####################### Authors: Yehia Ehab                        #######################
####################### Date : 10/06/2024                          #######################
##########################################################################################

# GUI Main #

# Importing PyQT packages
from PyQt5.QtWidgets import QApplication,QWidget,QMessageBox
from PyQt5.QtCore import Qt,QTimer
from PyQt5.QtGui import QPixmap,QImage
from PyQt5.uic import loadUiType

# Importing required packages
import ntpath, sys, os

# Importing 
import cam

# Load UI
FormClass, _ = loadUiType(ntpath.join(os.getcwd(), './UI/main.ui'))

# Define main window
class InmoovArm (QWidget, FormClass):
    def __init__(self):
        super(InmoovArm, self).__init__()
        QWidget.__init__(self)
        self.setupUi(self)
        self.Handle_Buttons()   # Handle GUI Buttons
                
        # Camera 1 Thread
        self.Camera1 = cam.Camera(self)
        self.Camera1.imageUpdate.connect(self.Handle_UpdateCam1)
        self.Camera1.start()

    # GUI buttons
    def Handle_Buttons(self):
        self.hand.clicked.connect(self.Handle_HandDetection)               # Hand Detection Button
        self.face.clicked.connect(self.Handle_FaceDetection)               # Face Detection Button
        self.currency.clicked.connect(self.Handle_CurrencyDetection)       # Currency Detection Button
        self.rotateR.clicked.connect(self.Handle_RotateR)                  # Rotate Right Function            
        self.rotateL.clicked.connect(self.Handle_RotateL)                  # Rotate Left Function                     
        self.togCam.stateChanged.connect(self.Handle_ToggleCamera)         # Toggle Camera 
        self.exit.clicked.connect(self.Handle_Exit)                        # Exit Button         

    # Toggle Camera Visibility
    def Handle_ToggleCamera(self,state):
        if state==0:
            self.Camera1.start()
        else:
            self.cam1.setPixmap(QPixmap('./Images/Solid_black.png'))
            self.Camera1.terminate()   

    def Handle_RotateR(self):
        self.Camera1.rotate('right')

    def Handle_RotateL(self):
        self.Camera1.rotate('left')

    # Camera Feed Function
    def Handle_UpdateCam1(self,pic):
        ConvertToQtFormat = QImage(pic.data, pic.shape[1], pic.shape[0], QImage.Format_RGB888)
        Pic = ConvertToQtFormat.scaled(self.cam1.width(), self.cam1.height())
        self.cam1.setPixmap(QPixmap.fromImage(Pic))  # Update Label

    # Hand Detection Button
    def Handle_HandDetection(self):
        self.detected.setText('')
        self.number.setText('')
        self.detection.setText('Hand:  ')
        self.data.setText('Fingers:  ')
        self.Camera1.setHandsMode()

    # Face Detection Button
    def Handle_FaceDetection(self):
        self.detected.setText('')
        self.number.setText('')
        self.detection.setText('Person:  ')
        self.data.setText('Probability:  ')
        self.Camera1.setFaceMode()
    
    # Currency Detection Button
    def Handle_CurrencyDetection(self):
        self.detected.setText('')
        self.number.setText('')
        self.detection.setText('Currency:  ')
        self.data.setText('Probability:  ')
        self.Camera1.setCurrencyMode()
    
    # Exit Button
    def Handle_Exit(self):
        # Kill all threads
        self.Camera1.terminate()
        self.close()

# Reading Scaling Txt File
with open('./UI/scaling.txt','r') as f:
    for line in f.readlines():
        if line[0]=='G':
            scalingFlag = int(line.split('=')[1])

# GUI Scaling
if scalingFlag==1:
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

# Executing GUI
if __name__ == '__main__':
    app = QApplication(sys.argv)
    Window_Loop = InmoovArm()
    Window_Loop.show()
    app.exec()