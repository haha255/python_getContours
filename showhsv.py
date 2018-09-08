import cv2
import numpy as np
from matplotlib import pyplot as plt
image=cv2.imread('s6.jpg')
HSV=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
def getpos(event,x,y,flags,param):
    if event==cv2.EVENT_LBUTTONDOWN:
        print(HSV[y,x])
#th2=cv2.adaptiveThreshold(imagegray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
cv2.imshow("imageHSV",HSV)
cv2.imshow('image',image)
cv2.setMouseCallback("imageHSV",getpos)
cv2.waitKey()
'''
import cv2
import numpy as np

cap = cv2.VideoCapture(1)
# image=cv2.imread("./src/7.png")
while (1):
    ret, image = cap.read()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower = np.array([100, 60, 100])
    upper = np.array([120, 120, 180])

    mask = cv2.inRange(hsv, lower, upper)
    res = cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow('image', image)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    # cv2.waitKey(0)
    k = cv2.waitKey(5) & 0xff
    if k == 27:
        break
cv2.destroyAllWindows()'''