from turtle import pensize
import numpy as np
import cv2 as cv 
import json


source = 1 #0 is native, 1 is external webcam
camera = cv.VideoCapture(source, cv.CAP_DSHOW)
camera.set(cv.CAP_PROP_EXPOSURE, -7) # change to -1 for internal camera, -7 for FISHEYE, -4 for Microsoft hd3000


while True:
    ret, img = camera.read()
    
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(img, (5, 5), 0)
    canny = cv.Canny(blur, 100, 70)
    ret, mask = cv.threshold(canny, 70, 255, cv.THRESH_BINARY)

    cv.imshow('Video feed', mask)
    cv.imshow("Live video", img)
    
    if cv.waitKey(1) == ord('q'):
        break