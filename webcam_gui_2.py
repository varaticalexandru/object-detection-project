import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2 as cv

# main window object
class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()

        # thread within main program loop
        self.Worker1 = Worker1()

        # layout
        self.VBL = QVBoxLayout()

        # label
        self.FeedLabel = QLabel()
        self.VBL.addWidget(self.FeedLabel)

        # button
        self.CancelBTN = QPushButton("Cancel")
        self.CancelBTN.clicked.connect(self.CancelFeed)

        self.StartBTN = QPushButton("Start")
        self.StartBTN.clicked.connect(self.Worker1.start)

        self.VBL.addWidget(self.StartBTN)
        self.VBL.addWidget(self.CancelBTN)

        # connect the signal to a slot
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)

        self.setLayout(self.VBL)

    # slot (function)
    def ImageUpdateSlot(self, Image):
        self.FeedLabel.setPixmap(QPixmap.fromImage(Image))
        
    def CancelFeed(self):
        self.Worker1.stop()
    

# qthread class
# retrieve img & convert it & send signal
class Worker1(QThread):

    # signal
    ImageUpdate = pyqtSignal(QImage)
    
    def run(self):

        self.ThreadActive = True
        capture = cv.VideoCapture(0)

        if not capture.isOpened():
            print("Error. Could not open the VideoCapture.")
            sys.exit(1)
        
        while self.ThreadActive:
            isTrue, frame = capture.read()
            if isTrue:
                image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                flipped = cv.flip(image, 1) # flip on vertical axis
                qimage = QImage(flipped.data, flipped.shape[1], flipped.shape[0], QImage.Format_RGB888)
                resized = qimage.scaled(640, 480, Qt.KeepAspectRatio)
                self.ImageUpdate.emit(resized)

        capture.release()


    def stop(self):

        self.ThreadActive = False

        self.quit()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    root = MainWindow()
    root.show()
    sys.exit(app.exec())