import cv2 
import numpy as np
import time  

# declaration of variables  
initialTime = 0  
initialDistance = 0  
changeInTime = 0  
changeInDistance = 0  
  
listDistance = []  
listSpeed = []

def empty(a):
    pass

video = cv2.VideoCapture(0)
video.set(3,640)
video.set(4,480)
video.set(10,100)

font = cv2.FONT_HERSHEY_COMPLEX 

cv2.namedWindow("TrackBar")
cv2.resizeWindow("TrackBar", 640, 240)
cv2.createTrackbar("Hue Min","TrackBar",85,179,empty)
cv2.createTrackbar("Hue Max","TrackBar",94,179,empty)
cv2.createTrackbar("Sat Min","TrackBar",54,255,empty)
cv2.createTrackbar("Sat Max","TrackBar",150,255,empty)
cv2.createTrackbar("Val Min","TrackBar",130,255,empty)
cv2.createTrackbar("Val Max","TrackBar",196,255,empty)

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters",640,240)
cv2.createTrackbar("Threshold1","Parameters",177,255,empty)
cv2.createTrackbar("Threshold2","Parameters",27,255,empty)
cv2.createTrackbar("Area","Parameters",1500,30000,empty)

def getDistance(img,imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areaMin = cv2.getTrackbarPos("Area", "Parameters")
        if area > areaMin:
            distance = 2*(10**(-7))* (area**2) - (0.0067 * area) + 83.487
            M = cv2.moments(cnt)
            Cx = int(M['m10']/M['m00'])
            Cy = int(M['m01'] / M['m00'])
            # S = 'Location of object:' + '(' + str(Cx) + ',' + str(Cy) + ')'
            # cv2.putText(imgContour, S, (5, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            # S = 'Area of contour: ' + str(area)
            # cv2.putText(imgContour, S, (5, 100), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            S = 'Jarak Objek : ' + str(distance)
            cv2.putText(imgContour, S, (5, 50), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.drawContours(imgContour, cnt, -1, (0, 255, 0), 3)

def speedFinder(distance, timeTaken):
    speed = coveredDistance / timeTaken  
    return speed

def averageFinder(completeList, averageOfItems):  
    lengthOfList = len(completeList)  
    selectedItems = lengthOfList - averageOfItems  
    selectedItemsList = completeList[selectedItems:]  
    average = sum(selectedItemsList) / len(selectedItemsList)  
  
    return average

while True:
    success, imgN = video.read()
    img = cv2.resize(imgN,(720,540))
    img = cv2.flip(img, +1) 
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    imgContour = img.copy()

    h_min = cv2.getTrackbarPos("Hue Min", "TrackBar")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBar")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBar")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBar")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBar")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBar") 

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHSV, lower, upper)

    result = cv2.bitwise_and(img,img, mask = mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    imgBlur = cv2.GaussianBlur(result, (7, 7), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
    imgCanny = cv2.Canny(imgGray, threshold1, threshold2)
    kernel = np.ones((7, 7))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
    getDistance(imgDil, imgContour)
    
    # if initialDistance != 0:
    #     # getting the  difference of the distances  
    #     changeInDistance = initialDistance - distance
    #     changeInTime = time.time() - initialTime
  
    #     # calculating the speed  
    #     speed = speedFinder(changeInDistance, changeInTime)  
    #     listSpeed.append(speed) 
    #     averageSpeed = averageFinder(listSpeed, 10)  
    #     if averageSpeed < 0:  
    #         averageSpeed = averageSpeed * -1  
        
    #     # filling the progressive line dependent on the speed.  
    #     speedFill = int(45+(averageSpeed) * 130)  
    #     if speedFill > 235:
    #         speedFill = 235
    
    # cv2.putText(imgContour, f"Speed: {round(averageSpeed, 2)} m/s", (50, 75), fonts, 0.6, (0, 255, 220), 2)

    cv2.imshow("Bitwise",result)
    cv2.imshow('Hasil', imgContour)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break