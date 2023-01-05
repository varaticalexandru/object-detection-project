import os
import warnings
import cv2 as cv
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from PIL import Image

warnings.filterwarnings('ignore')

# from helper_functions import run_odt_and_draw_results
import config

cwd = os.getcwd()

MODEL_PATH = config.MODEL_PATH
MODEL_NAME = config.MODEL_NAME

DETECTION_THRESHOLD = 0.3

# Change the test file path to your test image
# INPUT_IMAGE_PATH = r'C:\Users\Admin\Pictures\cards_dataset_angle\test\img_161.jpg'

capture = cv.VideoCapture(0)
# capture.set(cv.CAP_PROP_FPS, 30)
# capture.set(3, 416) #width
# capture.set(4, 416) #height



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
imW = 416
imH = 416

labels = ['card']


while True:

    isTrue, frame = capture.read()
    
    # resize frame to imW, imH
    frame = cv.resize(frame, (imW, imH))

    if cv.waitKey(20) & 0xFF == ord('e') or not isTrue:
        print("Video ended")
        break

    # convert to RGB, resize input to model expected input
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame_resized = cv.resize(frame_rgb, (320, 320))

    # convert frame to expected shape
    input_data = np.expand_dims(frame_resized, axis = 0)

    # perform actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # retrieve detection results
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects

    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            cv.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv.FILLED) # Draw white box to put label text in
            cv.putText(frame, label, (xmin, label_ymin-7), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
    

    cv.imshow("Video", frame)
    
capture.release()
cv.destroyAllWindows()

print("Program done")


# im = Image.open(INPUT_IMAGE_PATH)
# im.thumbnail((512, 512), Image.ANTIALIAS)
# im.save(f'{cwd}/result/input.png', 'PNG')



# # Run inference and draw detection result on the local copy of the original file
# detection_result_image = run_odt_and_draw_results(
#     f'{cwd}/result/input.png',
#     interpreter,
#     threshold=DETECTION_THRESHOLD
# )

# # Show the detection result
# img = Image.fromarray(detection_result_image)
# img.save(f'{cwd}/result/ouput.png')
# print('-'*100)
# print('See the result folder.')