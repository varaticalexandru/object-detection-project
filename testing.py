# 0. import libraries that do not use multithreading:
import os, sys, time
from multiprocessing import Process, Pipe
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

class VideoProcess(Process):

    def __init__(self):
        super().__init__()
        self.front_pipe, self.back_pipe = Pipe()

    # BACKEND

    def run(self):

        self.capture = cv.VideoCapture(0)
        if not self.capture.isOpened():
            print("Error opening video capture")
            sys.exit(1)

        print("Success opening video capture")

        self.active = True
        while True:
            #time.sleep(1)
            self.active, frame = self.capture.read()
            self.back_pipe.send(frame)
            print("sent frame")


    # FRONTEND

    def getPipe(self):
        return self.front_pipe

    def getFrame(self):
        return self.front_pipe.recv()


    def stop(self):
        self.front_pipe.send("stop")
        self.join()
        

if __name__ == "__main__":
    p = VideoProcess()
    p.start()

    pipe = p.getPipe()

    while True:
        #time.sleep(1)
        frame = p.getFrame()
        print("received frame")
        cv.imshow("frame", frame)
        cv.waitKey(20)

    print("Program done")