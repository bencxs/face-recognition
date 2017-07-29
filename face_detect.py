from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import datetime
import time
import random
import string
import threading

from model import Model

global preds

# path to load trained model
SAVE_PATH = 'checkpoints/convnet_face/face-convnet-2'

# load neural net model
face_rcog = Model()

# load Haar Cascade for face detection
detector= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

def get_focal_length(known_distance, known_width, pixel_width):
    '''
    calculate camera's focal length
    '''
    f = (pixel_width * known_distance) / known_width
    return f

def distance_to_camera(known_width, focal_length, pixel_width):
    '''
    calculate distance of object to camera using the triangular approximation
    '''
    d = (known_width * focal_length) / pixel_width
    return int(round(d, 0))

def get_face_img(faces, img, size, save_img=False):
    '''
    outputs face boundaries to array
    '''
    subjects = dict()
    for i, (x, y, w, h) in enumerate(faces):
        roi = img[y: y + h, x: x + w]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi = cv2.resize(roi, (size, size))
        subjects[i] = roi
        if save_img == True:
            cv2.imwrite("data/" + str(i) + "_" + str(x) + "_" + str(y) + ".png", roi)
    return [v for k, v in subjects.items()]


# calibrate camera parameters
known_width = 20 #cm
known_distance = 29.7 #cm
pixel_width = 305 #pixels

# counter to keep track of prediction delay time
counter = 0

while(True):
    ret, img = cap.read()
    height, width, channels = img.shape
    # print height, width, channels
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)

    # capture faces to array
    face_snapshots = get_face_img(faces, img, 64, save_img=False) # output to 64x64 pixels

    # print(len(face_snapshots))

    # introduce a delay in predicting faces
    if counter % 100 == 0:
        # make face prediction using neural net
        try:
            preds = face_rcog.predict(face_snapshots, save_path=SAVE_PATH)
            print(preds)
        except:
            # if face input doesnt exist, output "None"
            print(["None"])
    counter += 1

    for i, (x, y, w, h) in enumerate(faces):
        # set parameters for fancy bounding box
        box_offset_width = int(round(0.33 * w, 0)) # 33% of width
        box_offset_height = int(round(0.33 * h, 0)) # 33% of height
        line_thickness = 2
        line_color = (0, 0, 255) #BGR

    	# draw regular bounding box over face
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # draw fancy bounding box 
        # top horizontal
        cv2.line(img, (x, y), (x + box_offset_width, y), 
            line_color, line_thickness)
        cv2.line(img, (x + w, y), (x + w - box_offset_width, y), 
            line_color, line_thickness)
        # bottom horizontal
        cv2.line(img, (x, y + h), (x + box_offset_width, y + h), 
            line_color, line_thickness)
        cv2.line(img, (x + w, y + h), (x + w - box_offset_width, y + h), 
            line_color, line_thickness)
        # left vertical
        cv2.line(img, (x, y), (x, y + box_offset_height), 
            line_color, line_thickness)
        cv2.line(img, (x, y + h), (x, y + h - box_offset_height), 
            line_color, line_thickness)
        # right vertical
        cv2.line(img, (x + w, y), (x + w, y + box_offset_height), 
            line_color, line_thickness)
        cv2.line(img, (x + w, y + h), (x + w, y + h - box_offset_height), 
            line_color, line_thickness)

        # draw label name below the face
        cv2.rectangle(img, (x, y + h), (x + w, y + h + 35), line_color, -1)
        font = cv2.FONT_HERSHEY_DUPLEX
        try:
            cv2.putText(img, str(preds[i]), (x + 6, y + h + 50), font, 1.0, (255, 255, 255), 1)
        except:
            # if no prediction present, put a placeholder first
            cv2.putText(img, str(""), (x + 6, y + h + 50), font, 1.0, (255, 255, 255), 1)

        # calculate distance from camera to face
        focal_length = get_focal_length(known_distance, known_width, pixel_width)
        face_distance = distance_to_camera(known_width, focal_length, w)
        cv2.putText(img, "Dist: " + str(face_distance) + "cm", 
            (x + 6, y + h + 25), font, 1.0, (255, 255, 255), 1)

    # show timestamp
    timestamp = datetime.datetime.now()
    ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
    cv2.putText(img, ts, (10, height - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

    # fancy text scanning
    s = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10)) # random 10 character string
    cv2.putText(img, "Scanning: " + s, (10, 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('frame', img)
    # press 'q' to quit window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()