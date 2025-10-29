#taffic monitoring and license plate detection
# 1. process the video of traffic
# 2. detect traffic light whether its red, yellow , green 
# 3. lane marking(white marks) to track vehicle movement
# 4. vehicle license plate is detected using the haar cascade detection 
# 5. OCR -> Extracting the character from the license plate
# 6. logs violation -> mysql database
# 7. Display the processed video with annotations

import cv2
import numpy as np 
import matplotlib.pyplot as plt
import os
import dbconnection
import re,pymysql
from PIL import Image
from collections import deque 
import pytesseract  # OCR (OPTICAL CHARACTER RECOGNISTION)
import easyocr




#Define the license plate cascade
license_plate_cascade = cv2.CascadeClassifier(" ")

#ensure the file exists and is loaded correctly
if license_plate_cascade.empty():
    raise FileNotFoundError("Haar Cascade for license plate detection not found. Ensure the path is correct.")

def detect_traffic_color(image, rect):
    x,y,w,h  = rect
    # Extract region of Interest (ROI) from the image based on the rectangle

    roi = image[y:y+h,x:x+w]

    #convert ROI to HSV color space #HSV -hue saturation Value

    hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    
    #Define HSV range for red color
    red_lower = np.array([0,120,70])
    red_upper = np.array([10,255,255])

    #Define HSV range for yellow color
    yellow_lower =np.array([20,100,100])
    yellow_upper = np.array([30,255,255])




