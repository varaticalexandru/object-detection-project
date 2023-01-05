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
        self.startWebcam.clicked.connect(self.display)
        self.stopWebcam.clicked.connect(self.stop)

    def display(self):
        # video capture (stream) object
        capture = cv.VideoCapture(0)

        if not capture.isOpened():
            print("error")

        # set the image format (RGB 256)
        qformat = QImage.Format_RGB888

        while True:
            isTrue, frame = capture.read()

            if cv.waitKey(20) & 0xFF == ord('e') or not isTrue:
                print("Video ended")
                break
    
            # resize frame to imW, imH
            frame_resized = cv.resize(frame, (640, 480))

            # create QImage (for QLabel)
            outFrame = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], qformat)

            # display image on QLabel
            self.videoLabel.setPixmap(QPixmap.fromImage(outFrame))
            self.videoLabel.setScaledContents(True)

            
        capture.release()
        cv.destroyAllWindows()

    def stop(self):
        pass
        


if __name__ == '__main__':

    app = QApplication(sys.argv)
    window = Window()
    window.setWindowTitle("App")
    window.show()
    sys.exit(app.exec())
