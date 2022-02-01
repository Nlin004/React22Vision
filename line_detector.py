from turtle import pensize
import numpy as np
import math
import cv2 as cv 
import json


source = 1 #0 is native, 1 is external webcam
camera = cv.VideoCapture(source, cv.CAP_DSHOW)
camera.set(cv.CAP_PROP_EXPOSURE, -7) # change to -1 for internal camera, -7 for FISHEYE, -4 for Microsoft hd3000
MIN_AREA = 500

def sobel_edge(frame):
    depth = cv.CV_16S
    scale = 1
    delta = 0

    frame = cv.GaussianBlur(frame, (3,3), 0)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    grad_x = cv.Sobel(gray, depth, 1,0,ksize=3, scale = scale, delta = delta, borderType=cv.BORDER_DEFAULT)
    grad_y = cv.Sobel(gray, depth, 0, 1, ksize=3, scale = scale, delta = delta, borderType=cv.BORDER_DEFAULT)
    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)
    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return grad

def laplace_edge(frame):
    ddepth = cv.CV_16S
    kernel_size = 3
    src = cv.GaussianBlur(frame, (3,3), 0)
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    # dst = cv.adaptiveThreshold(src_gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
    # dst  = cv.Canny(src_gray, 50, 150, apertureSize=3)
    dst = cv.Laplacian(src_gray, ddepth, ksize=kernel_size)
    abs_dst = cv.convertScaleAbs(dst)
    return abs_dst

def isRect(cnt, approx, ar):
    return len(approx) == 4 and cv.contourArea(cnt) > MIN_AREA and not (0.8 <= ar <= 1.1)

def maskColor(frame):
    # frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    # lower_gray = np.array([0, 0, 0], np.uint8)
    # upper_gray = np.array([40, 58, 48], np.uint8)
    # # mask = cv.InRangeS(imgHSV, cv.Scalar(0, 0, 0, 0), cv.Scalar(180, 255, 30, 0), imgThreshold)
    # # thresh = cv.threshold(frame, 60, 255, cv.THRESH_BINARY_INV)
    # mask_gray = cv.inRange(frame, lower_gray, upper_gray)
    # # img_res = cv.bitwise_and(mask_gray, mask_gray, mask = mask_gray)

    # # mask = cv.inRange(frame, (0,0,0), (90,129,150))
    # frame = cv.bitwise_and(frame, frame, mask = mask_gray)
    # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # return frame # frame 

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5,5), 3)
    ret, thresh = cv.threshold(blurred, 45, 255, cv.THRESH_BINARY_INV)
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, (13,13))
    # thresh = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)
    return thresh

def findRect(frame, output):
    contours, _ = cv.findContours(frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for(index, contour) in enumerate(contours):
        peri = cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, 0.01 * peri, True)
        x,y,w,h = cv.boundingRect(approx)
        aspect_ratio = w / h
        
        if isRect(contour, approx, aspect_ratio):
            cv.rectangle(output, (x,y), (x+w, y+h), (0,255,255))

            _, cols = frame.shape[:2]
            [vx, vy,x,y] = cv.fitLine(contour, cv.DIST_L2,0,0.01,0.01)
            lefty = int((-x*vy/vx) + y)
            righty = int(((cols-x)*vy/vx)+y)
            p1 = (cols-1, righty)
            p2 = (0, lefty)

            
            cv.line(output, p1,p2,(0,0,255),2)
            print(get_angle(p1, p2))
            cv.putText(output, f"{str(get_angle(p1,p2))} degrees", (int(x), int(y-10)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255)) 

def get_angle(p1,p2):
    
    m = ((p1[1] - p2[1]) / (p1[0] - p2[0]))
    return format(math.atan(m) * -180 / math.pi, '.0f')
         

def detect_line(frame):
    masked_frame = maskColor(frame)
    findRect(masked_frame, frame)

    cv.imshow("Line Mask", masked_frame)

    if cv.waitKey(1) & 0xFF is ord('q'):
        return frame


    return frame


# while True:
#     ret, img = camera.read()
    
#     # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     blur = cv.GaussianBlur(img, (5, 5), 0)
#     canny = cv.Canny(blur, 100, 70)
#     ret, mask = cv.threshold(canny, 70, 255, cv.THRESH_BINARY)

#     cv.imshow('Video feed', mask)
#     cv.imshow("Live video", img)
    
#     if cv.waitKey(1) == ord('q'):
#         break