import cv2 
import numpy as np

def empty(a):
    pass

video = cv2.VideoCapture(0)
video.set(3,640)
video.set(4,480)
video.set(10,100)

cv2.namedWindow("TrackBar")
cv2.resizeWindow("TrackBar", 640, 240)
cv2.createTrackbar("Hue Min","TrackBar",5,179,empty)
cv2.createTrackbar("Hue Max","TrackBar",19,179,empty)
cv2.createTrackbar("Sat Min","TrackBar",119,255,empty)
cv2.createTrackbar("Sat Max","TrackBar",242,255,empty)
cv2.createTrackbar("Val Min","TrackBar",152,255,empty)
cv2.createTrackbar("Val Max","TrackBar",255,255,empty)

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters",640,240)
cv2.createTrackbar("Threshold1","Parameters",177,255,empty)
cv2.createTrackbar("Threshold2","Parameters",27,255,empty)
cv2.createTrackbar("Area","Parameters",420,30000,empty)

def getContours(img,imgContour):
    global dir
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areaMin = cv2.getTrackbarPos("Area", "Parameters")
        if area > areaMin:
            # cv2.drawContours(imgContour, cnt, -1, (0, 255, 255), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            #print(len(approx))
            x , y , w, h = cv2.boundingRect(approx)
            wide = w*h
            print("x = ",str(x),"\n","y = ",str(y),"\n","w = ",str(w),"\n","h = ",str(h),"\n","\n")
            cv2.rectangle(imgContour,(x,y),(x+w,y+h),(255,0,0),2)
            center = (x+(w//2)-10,y+(h//2)+6)
            upper = (x+(w//2)-30,y+(h//2)-20)
            cv2.putText(imgContour, f'Wide : {int(wide)}', (50, 100), cv2.FONT_HERSHEY_PLAIN, 5,(255, 0, 0), 5)
            cv2.putText(imgContour,"+",center,cv2.FONT_HERSHEY_COMPLEX,0.9,(0,255,255),2)
            # cv2.putText(imgContour,"MATANG",upper,cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),2)

while True:
    success, imgN = video.read()
    img = cv2.resize(imgN,(720,540))
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
    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
    getContours(imgDil, imgContour)
    cv2.imshow("Bitwise",result)
    cv2.imshow('Hasil', imgContour)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break