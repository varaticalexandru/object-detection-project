import sys, os, logging, time
from datetime import datetime
from multiprocessing import Process, Pipe

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.uic import loadUi

import tensorflow as tf
import numpy as np
import cv2 as cv

cwd = os.getcwd()

MODEL_PATH = f'{cwd}/model'
MODEL_NAME = 'model_1.tflite'

model_path = f'{MODEL_PATH}/{MODEL_NAME}'

boxes_idx, classes_idx, scores_idx = 1, 3, 0

min_conf_threshold = 0.6 # minimum confidence threshold
imW = 640 # image width
imH = 480 # image height

labels = ['card'] # classes



# video process class
class VideoProcess(Process):

    def __init__(self):
        super().__init__()
        self.front_pipe, self.back_pipe = Pipe() # open pipe

    # BACKEND

    def run(self):

        self.capture = cv.VideoCapture(0)
        if not self.capture.isOpened():
            print("Error opening video capture")
            sys.exit(1)

        print("Success opening video capture")

        self.active = True
        while self.active:
            isTrue, frame = self.capture.read()
            if not isTrue:
                self.terminate()
            self.back_pipe.send(frame) # send frame from backend


    # FRONTEND

    def getFrame(self):
        return self.front_pipe.recv() # receive frame to frontend

    def getPipe(self):
        return self.front_pipe

    def stop(self):
        self.front_pipe.send("stop")
        self.join()


# QT dialog window class
class Window(QDialog):
    
    def __init__(self, pipe):
        super(Window, self).__init__()
        
        # target number of detected objects (input)
        self.validate = 0
        self.target = 0


        # load ui
        loadUi('app.ui', self)
        self.numberText.setText("0")
        self.targetEdit.setText("0")

        # thread
        self.Worker1 = Worker1(pipe)

        # buttons
        self.startWebcam.clicked.connect(self.Start)
        self.stopWebcam.clicked.connect(self.CancelFeed)
        self.valButton.clicked.connect(self.Validate)

        # connect signal to slot
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)
    
    # start
    def Start(self):
        self.Worker1.start()
        self.logEdit.append(f"Success opening video capture")
        self.logEdit.append(f"INFO: Created TensorFlow Lite XNNPACK delegate for CPU.\n")


    # validate
    def Validate(self):
        self.target = int(self.targetEdit.toPlainText())    # read input number of detections
        self.validate = 1

    # slot (function)
    def ImageUpdateSlot(self, Coords, Frame, Image, DetectionsNumber):
        self.videoLabel.setPixmap(QPixmap.fromImage(Image))
        self.numberText.setText(f"{DetectionsNumber}")

        # save capture/frame if number of detections = target
        if self.validate and self.target and self.target == DetectionsNumber:
            if "captures" not in os.listdir('.') and not os.path.isfile("captures"):
                os.mkdir("captures")

            name = r'{}'.format("captures/" + datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + ".jpg")
            #print(f"Capture successfully saved: {name}")
            self.logEdit.append(f"Match : Capture successfully saved: {name}")
            cv.imwrite(name, cv.cvtColor(Frame, cv.COLOR_BGR2RGB))

        # debug if number of detections != target
        elif self.validate and self.target and self.target != DetectionsNumber:
            if "logs" not in os.listdir('.') and not os.path.isfile("logs"):
                os.mkdir("logs")

            name = r'{}'.format("logs/" + datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + ".log")
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s : %(levelname)s : %(message)s',
                filename=name,
                filemode='w',
            )

            logging.debug(f'Mismatch : Got {DetectionsNumber} detections : Expected {self.target} detections : Bounding box coordinates of detected objects {Coords}')
            #print(f"Logged successfully: {name}")
            self.logEdit.append(f"Mismatch : Logged successfully: {name}")

        self.validate = 0
        self.target = 0

    # cancel
    def CancelFeed(self):
        self.Worker1.stop()
        self.logEdit.append(f"Video stopped")


# qthread class
# retrieves img & converts it & processes it & sends signal back
class Worker1(QThread):

    def __init__(self, pipe):
        super(Worker1, self).__init__()
        self.pipe = pipe


    # signal
    ImageUpdate = pyqtSignal(list, np.ndarray, QImage, int)

    def run(self):

        # Load the TFLite model
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        # load the input shape required by the model
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        # input_height = input_details[0]['shape'][1]
        # input_width = input_details[0]['shape'][2]

        self.ThreadActive = True
        capture = cv.VideoCapture(0)

        if not capture.isOpened():
            print("Error. Could not open the VideoCapture.")
            #self.logEdit.append(f"Error. Could not open the VideoCapture.")
            sys.exit(1)
        
        while self.ThreadActive:
            frame = self.pipe.recv()    # receive frame from pipe's frontend
            if True:
                frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB) # BGR >> RGB
                frame = cv.resize(frame_rgb, (imW, imH))
                frame_resized = cv.resize(frame, (320, 320)) # resize to model shape

                input_data = np.expand_dims(frame_resized, axis=0)  # convert to batch shape

                # perform actual detection by running the model with the image as input
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()

                # retrieve detection results
                boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
                classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
                scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects

                n = len([x for x in scores if x > min_conf_threshold])  # number of detected objects
                coords = []

                for i in range(len(scores)):
                    if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                        # Get bounding box coordinates and draw box
                        # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                        ymin = int(max(1,(boxes[i][0] * imH)))
                        xmin = int(max(1,(boxes[i][1] * imW)))
                        ymax = int(min(imH,(boxes[i][2] * imH)))
                        xmax = int(min(imW,(boxes[i][3] * imW)))

                        coords.append([xmin, ymin, xmax, ymax])
                        
                        cv.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                        # Draw label
                        object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                        label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                        label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                        cv.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv.FILLED) # Draw white box to put label text in
                        cv.putText(frame, label, (xmin, label_ymin-7), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

                qimage = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
                resized = qimage.scaled(imW, imH, Qt.KeepAspectRatio)
                
                # send/emit signal back to main GUI thread
                self.ImageUpdate.emit(coords, frame, resized, n)

        capture.release()


    def stop(self):
        self.ThreadActive = False
        self.quit()


if __name__ == '__main__':

    # start video process       
    p = VideoProcess()
    p.start()
    pipe = p.getPipe()

    app = QApplication(sys.argv)
    window = Window(pipe)
    window.show()
    window.setWindowTitle("Image Processing App")

    print("Program done")
    sys.exit(app.exec())