from calendar import c
import cv2 as cv
from cv2 import cvtColor
import numpy as np
import json



f = open("data.json")
data = json.load(f)

#accessing camera
source = 1 #0 is native, 1 is external webcam
camera = cv.VideoCapture(source, cv.CAP_DSHOW)
camera.set(cv.CAP_PROP_EXPOSURE, -4) # change to -1 for internal camera, -7 for FISHEYE, -4 for Microsoft hd3000


def getDistance(focal_length, real_width, width_in_frame): # FIX THIS
    distance = (real_width * focal_length) / width_in_frame
   
    return distance
   

def isCircle(cnt, contours):
    approx = cv.approxPolyDP(cnt, 0.01 * cv.arcLength(cnt, True), True)
    
    (coord_x, coord_y), radius = cv.minEnclosingCircle(cnt)
    #center = (int(coord_x), int(coord_y))    

    # if  1.0 >= cv.contourArea(cnt) / (radius**2 * 3.14) >= .8 and len(approx) > 8: 
    if len(approx) > 8 and (3.14 * cv.minEnclosingCircle(cnt)[1] ** 2 - cv.contourArea(cnt) < (3.14 * cv.minEnclosingCircle(cnt)[1] ** 2) * (1 - 0.75)):
        return True
            
    
    return False

def drawRect(mask, img, color):
    #color dictionary
    colors = {"red": (0, 0, 255), "blue": (255, 0, 0), "white": (255,255,255)}
    # mask = cv.bitwise_and(img, img, mask = mask)
    # mask = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
    # mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    
    colored_box = colors["white"]
    
    #Get contours on the mask
    contours = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) == 2:
        contours = contours[0]
        
    else:
        contours = contours[1]
    
    for contour in contours:
        (coord_x, coord_y), radius = cv.minEnclosingCircle(contour)
        center = (int(coord_x), int(coord_y))

        if isCircle(contour, contours):

            contour_area = cv.contourArea(contour)
            x, y, w, h = cv.boundingRect(contour)
            aspect_ratio = w/h
            #area = w * h

            #if  1.0 >= contour_area / (radius**2 * 3.14) >= .7 
            if .8 <= aspect_ratio <= 1.2 and contour_area > 350:
                #cv.circle(frame, (x+ (w/2), y+(y/2)), colored_box, 2)
                distance = getDistance(630, 24.13, int(w))
                distance = format((int(distance) * 1.1) / 100, '.2f')

                cv.circle(img, center, int(w/2), (0,255,0), 5)
                #cv.rectangle(frame, (x, y), (x + w, y + h), colored_box, 2)
                cv.putText(img, color.upper() + " BALL " + str(distance) + " METERS", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, colored_box, 2)
    #Draw rectangle with specific color using aspect ratio and area tests
    
          

def editImage(img):
    #blur, erode, and dilate frame
    
    img = cv.GaussianBlur(img, (11, 11), None)
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, (7,7))
    img = cv.morphologyEx(img, cv.MORPH_OPEN, (3,3))

    #img = cv.erode(img, None, iterations=2)
    #img = cv.dilate(img, None)
    
    return img


def createBlueMask(img):
    global mask_blue
    #blue values 
    #img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    lower1 = np.array(data[str(source)]["hsv_blue"]["lower"])
    upper1 = np.array(data[str(source)]["hsv_blue"]["upper"])

    # if(source == 1): #for fisheye:
    #     lower1 = np.array([76,105,17])               # S = 80 for pants!  76 80 17 my laptop cam: ([99,90,50]) # (45, 120, 50)
    #     upper1 = np.array([127,207,208]) # 127 207 208
    # elif(source == 0): # native cam:
    #     lower1 = np.array([87,71,1]) #([141,99,7])
    #     upper1 = np.array([123,255,255])# [217, 255, 255]                    # ([132,255,255]) #v (163, 255, 255)
    
    mask_blue = cv.inRange(hsv, lower1, upper1)
    # mask_blue = cv.morphologyEx(mask_blue, cv.MORPH_CLOSE, (7,7))
    # mask_blue = cv.cvtColor(mask_blue, cv.COLOR_HSV2BGR)
    # mask_blue = cv.cvtColor(mask_blue, cv.COLOR_BGR2GRAY)

    #mask_blue = cv.bitwise_and(img, img, mask = mask_blue)
    #mask_blue = cv.cvtColor(mask_blue, cv.COLOR_HSV2BGR)
   
    
    
    return mask_blue

def createRedMask(img):
    global mask_red
    #img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    #red values
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    
    lower1 = np.array(data[str(source)]["hsv_red1"]["lower"])
    upper1 = np.array(data[str(source)]["hsv_red1"]["upper"])
    lower2 = np.array(data[str(source)]["hsv_red2"]["lower"])
    upper2 = np.array(data[str(source)]["hsv_red2"]["upper"])
    
    if(source == 0): # integrated webcam
        #lower_red_2 = np.array([0,0,0])
        #upper_red_2 = np.array([157,255,255])

        #mask_red1 = cv.inRange(hsv, lower_red_1, upper_red_1)
        mask_red2 = cv.inRange(hsv, lower1, upper1)

        mask_red = cv.bitwise_not(mask_red2)
        mask_red = cv.morphologyEx(mask_red, cv.MORPH_CLOSE, (27, 27))
        #lower_red_1 = np.array([170, 85, 13])         #previous fisheye: ([135,120,31])         for my laptop camera: ([163, 49, 0])
        #upper_red_1 = np.array([229,255,255])                          #fisheye1: ([245,255,255])         ([245, 229, 255])
    elif(source == 1): #fisheye
        # lower_red_1 = np.array([135,120,31])   #np.array([0, 127,143])      #
        # upper_red_1 = np.array([245,255,255])  #np.array([94,255,255])      #np.array([245,255,255])  
        mask_red = cv.inRange(hsv, lower1, upper1)
        mask_red2 = cv.inRange(hsv, lower2, upper2)
        mask_red = mask_red + mask_red2
        #mask_red = cv.cvtColor(mask_red, cv.COLOR_HSV2BGR)
        #mask_red = cv.cvtColor(mask_red, cv.COLOR_BGR2GRAY)

        #mask_red = cv.bitwise_not(mask_red)
        # mask_red = cv.morphologyEx(mask_red, cv.MORPH_OPEN, (3,3))
    #mask_red = cv.bitwise_and(img, img, mask = mask_red)
    #mask_red = cv.cvtColor(mask_red, cv.COLOR_HSV2BGR)88

    return mask_red

#infinite loop to process live feed
def main(frame, message):
    # ret, frame = camera.read()
    
    #cv.normalize(frame, frame, 0, 255, cv.NORM_MINMAX)
    copy_frame = frame
    
    #call on functions
    if message == "red":
        drawRect( createRedMask(editImage(frame)), frame, "red")
    elif message == "blue":
        drawRect(    createBlueMask(editImage(frame)), frame, "blue")
    elif message == "both":
        drawRect( createRedMask(editImage(frame)), frame, "red")
        drawRect(    createBlueMask(editImage(frame)), frame, "blue")

    
    #show windows
    # cv.imshow("blue mask", createBlueMask(editImage(frame)))
    # cv.imshow("red mask", createRedMask(editImage(frame)))
    
    # # cv.imshow("frame", frame)
    # # cv.resizeWindow("frame", 640,480)

    # # #break loop if key pressed
    # if cv.waitKey(1) & 0xFF is ord('q'):
    #     return frame

    return frame

if __name__ == "__main__":
    main()

camera.release()
cv.destroyAllWindows()