import sys
import os

import cv2 as cv
from PyQt5.QtWidgets import QApplication, QDialog
from PyQt5.uic import loadUi
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# from helper_functions import run_odt_and_draw_results
import config

cwd = os.getcwd()

MODEL_PATH = config.MODEL_PATH
MODEL_NAME = config.MODEL_NAME

# Load the TFLite model
model_path = f'{MODEL_PATH}/{MODEL_NAME}'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# load the input shape required by the model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_height = input_details[0]['shape'][1]
input_width = input_details[0]['shape'][2]

boxes_idx, classes_idx, scores_idx = 1, 3, 0

min_conf_threshold = 0.5
imW = 640
imH = 480

labels = ['card']


# QT dialog window class
class Window(QDialog):

    def __init__(self):
        super(Window, self).__init__()

        # load ui
        loadUi('app.ui', self)
        self.numberText.setText("0")

        # frame (image) object
        self.frame = None

        # connect button clicking events to functions
        self.startWebcam.clicked.connect(self.update_frame)
        self.stopWebcam.clicked.connect(self.stop_webcam)

    def update_frame(self):

        # get opencv video capture object
        self.capture = cv.VideoCapture(0)

        # set capture width, height
        # self.capture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        # elf.capture.set(cv.CAP_PROP_FRAME_WIDTH, 640)

        if not self.capture.isOpened():
            print("Error initializing the video capture !\n")
            sys.exit()
        
        while True:
            # read the frame from capture object
            isTrue, self.frame = self.capture.read()

            if cv.waitKey(20) & 0xFF == ord('e') or not isTrue:
                print("Video ended")
                sys.exit()

            # convert read frame BGR >> RGB
            self.frame = cv.cvtColor(self.frame, cv.COLOR_BGR2RGB)

            # resize the frame
            self.frame = cv.resize(self.frame, (imW, imH))
            frame_resized = cv.resize(self.frame, (320, 320))

            # flip the frame
            # self.frame = cv.flip(self.frame, 1)

            # convert frame to expected shape
            input_data = np.expand_dims(frame_resized, axis = 0)

            # perform actual detection by running the model with the image as input
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            # retrieve detection results
            boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
            classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
            scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects

            # print number of detected objects
            self.numberText.setText(f"{len(scores)}")

            for i in range(len(scores)):
                n = len([x for x in scores if x > min_conf_threshold])
                self.numberText.setText(f"{n}")

                if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                    # Get bounding box coordinates and draw box
                    # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                    ymin = int(max(1,(boxes[i][0] * imH)))
                    xmin = int(max(1,(boxes[i][1] * imW)))
                    ymax = int(min(imH,(boxes[i][2] * imH)))
                    xmax = int(min(imW,(boxes[i][3] * imW)))
                    
                    cv.rectangle(self.frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                    # Draw label
                    object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                    label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv.rectangle(self.frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv.FILLED) # Draw white box to put label text in
                    cv.putText(self.frame, label, (xmin, label_ymin-7), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

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
        while True:
            pass
        


app = QApplication(sys.argv)
window = Window()
window.setWindowTitle("App")
window.show()
#cv.destroyAllWindows()
print("Program done")
sys.exit(app.exec())
