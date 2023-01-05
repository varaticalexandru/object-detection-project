import sys

import cv2 as cv
from PyQt5.QtWidgets import QApplication, QDialog
from PyQt5.uic import loadUi
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap



class Window(QDialog):

    def __init__(self):
        super(Window, self).__init__()

        # load ui
        loadUi('app.ui', self)

        # frame (image) object
        self.frame = None

        # connect button clicking events to functions
        self.startWebcam.clicked.connect(self.start_webcam)
        self.stopWebcam.clicked.connect(self.stop_webcam)

    def start_webcam(self):
        # get opencv video capture object
        self.capture = cv.VideoCapture(0)

        if not self.capture.isOpened():
            print("Error initializing the video capture !\n")
            sys.exit()

        # set capture width, height
        #self.capture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        #elf.capture.set(cv.CAP_PROP_FRAME_WIDTH, 640)

        # create the timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)

    def update_frame(self):
        # read the frame from capture object
        isTrue, self.frame = self.capture.read()

        # resize the frame
        self.frame = cv.resize(self.frame, (640, 480))

        # convert read frame BGR >> RGB
        self.frame = cv.cvtColor(self.frame, cv.COLOR_BGR2RGB)

        # flip the frame
        self.frame = cv.flip(self.frame, 1)

        # call display function
        self.display_frame(self.frame, 1)
            

    def display_frame(self, frame, window=1):

        # set the image format (RGB 256)
        qformat = QImage.Format_RGB888

        # create QImage (for QLabel)
        outFrame = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], qformat)

        # display image on QLabel
        self.videoLabel.setPixmap(QPixmap.fromImage(outFrame))
        self.videoLabel.setScaledContents(True)

       
    def stop_webcam(self):
        self.timer.stop()
        


if __name__ == '__main__':

    app = QApplication(sys.argv)
    window = Window()
    window.setWindowTitle("App")
    window.show()
    cv.destroyAllWindows()
    print("Program done")
    sys.exit(app.exec())
