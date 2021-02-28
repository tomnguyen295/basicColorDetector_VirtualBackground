import cv2 
import numpy as np
from util import stackImages

# Live testing HSV threshold values for video feed

def empty(x):
    pass



cv2.namedWindow("Trackbars") # Creating window for storing trackbars
cv2.resizeWindow("Trackbars", 640, 240) # Resizing window to appropriate size

### Creating trackbars ###
cv2.createTrackbar("Hue min", "Trackbars", 0, 179, empty) # Hue has 360 values but in openCV, we go 0-179 to fit the 8-bit range of 0-255
cv2.createTrackbar("Hue max", "Trackbars", 179, 179, empty)
cv2.createTrackbar("Sat min", "Trackbars", 0, 255, empty)
cv2.createTrackbar("Sat max", "Trackbars", 255, 255, empty)
cv2.createTrackbar("Val min", "Trackbars", 0, 255, empty)
cv2.createTrackbar("Val max", "Trackbars", 255, 255, empty)

cap = cv2.VideoCapture(0)

while True:
    ### Storing trackbar values in real time ###
    h_min = cv2.getTrackbarPos("Hue min", "Trackbars")
    h_max = cv2.getTrackbarPos("Hue max", "Trackbars")
    s_min = cv2.getTrackbarPos("Sat min", "Trackbars")
    s_max = cv2.getTrackbarPos("Sat max", "Trackbars")
    v_min = cv2.getTrackbarPos("Val min", "Trackbars")
    v_max = cv2.getTrackbarPos("Val max", "Trackbars")

    ### Setting boundaries for mask in real-time ##
    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])

    # upper = np.array([10,255,233]) # Detecting red apple

    ### Prepping images ##
    success, img = cap.read()
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # Converts RGB to HSV for color detection
    mask1 = cv2.inRange(imgHSV, lower, upper)
    imgOutput = cv2.bitwise_and(img, img, mask=mask1)
    # cv2.imshow("Original", img)
    # cv2.imshow("Apples", imgHSV)
    # cv2.imshow("Mask", mask)
    # cv2.imshow("Color Detection", imgOutput)
    imgCollage = stackImages(1, [img, imgOutput])
    cv2.imshow("Color Detecting", imgCollage)
    cv2.waitKey(1)
    




