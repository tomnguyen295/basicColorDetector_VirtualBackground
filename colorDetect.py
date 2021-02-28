import cv2
import numpy as np 
from util import stackImages

# We are going to make a program that takes an image and filters out all colors in that image except for our chosen color
# The result will be a 2x2 grid consisting of the original image, the image with only the selected color,
# the image with that color filtered out, and a final "green" screen, but this can be red, green, or blue
# Color choices may be updated later

file = input("Select an image to use: ")
color = input("What color do you want to detect? Choose from red, green, blue for now: ")
optional = input("Would you like to replace the color with a different background? Type y or n: ")


HSV_ranges = {'red': np.array([[0, 120, 70],[10,255,255]]),
            'green': np.array([[25, 55, 75],[102,255,255]]),
            'blue': np.array([[103,80,2],[125,255,255]])}

img = cv2.imread(file)
img = cv2.resize(img, (400,400))
imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # Converts RGB to HSV for more accurate color detection 
lower = HSV_ranges[color][0]
upper = HSV_ranges[color][1]
mask = cv2.inRange(imgHSV, lower, upper) # Creates a mask for isolating the chosen color

# Red values for H wrap around 0 and 180 so we need to add a second mask
if color == "red":
    lower2 = np.array([170, 120, 70])
    upper2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(imgHSV, lower2, upper2)
    mask += mask2

imgIsolated = cv2.bitwise_and(img, img, mask=mask) # Image with only the chosen color

antiMask = cv2.bitwise_not(mask) # Bitwise not operation on mask to create a reverse mask for filtering out the chosen color

imgFiltered = cv2.bitwise_and(img,img,mask=antiMask) # Image without the chosen color

# If user wants to see the optional virtual background
if optional == 'y':
    background = input("Choose a background image to replace the color with: ")
    background = cv2.imread(background)
    background = cv2.resize(background, (img.shape[1],img.shape[0])) # Resize background so that both are the same size
    bg_filtered = cv2.bitwise_and(background,background,mask=mask)
    gScreen = cv2.addWeighted(imgFiltered,1,bg_filtered,1,0)
else:
    gScreen = np.ones_like(img)

imgCollage = stackImages(1, [[img, imgIsolated], [imgFiltered, gScreen]])
cv2.imshow("Result", imgCollage)
cv2.waitKey(0)