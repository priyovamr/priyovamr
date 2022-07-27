import cv2
from cv2 import IMWRITE_JPEG_RST_INTERVAL
import numpy as np
import time
 
curveList = []
avgVal=7
pTime = 0
cTime = 0

cap = cv2.VideoCapture(0)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    #channel_count = img.shape[2]
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1,y1), (x2,y2), (0, 255, 0), thickness=3)

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img

def process(image):
    # print(image.shape)
    height = image.shape[0]
    width = image.shape[1]
    region_of_interest_vertices = [
        (0, height),
        (width/2, height/2),
        (width, height)
    ]
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(gray_image, 100, 120)
    cropped_image = region_of_interest(canny_image,
                    np.array([region_of_interest_vertices], np.int32),)
    lines = cv2.HoughLinesP(cropped_image,
                            rho=2,
                            theta=np.pi/180,
                            threshold=50,
                            lines=np.array([]),
                            minLineLength=40,
                            maxLineGap=100)
    if lines is None:
        image_with_lines = image
    else:
        image_with_lines = draw_the_lines(image, lines)
    return image_with_lines
 
def warpImg(img,points,w,h,inv = False):
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    if inv:
        matrix = cv2.getPerspectiveTransform(pts2, pts1)
    else:
        matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgWarp = cv2.warpPerspective(img,matrix,(w,h))
    return imgWarp
 
def nothing(a):
    pass

def initializeTrackbars(intialTracbarVals,wT=720, hT=540):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 400, 200)
    cv2.createTrackbar("Width Top", "Trackbars", intialTracbarVals[0],wT//2, nothing)
    cv2.createTrackbar("Height Top", "Trackbars", intialTracbarVals[1], hT, nothing)
    cv2.createTrackbar("Width Bottom", "Trackbars", intialTracbarVals[2],wT//2, nothing)
    cv2.createTrackbar("Height Bottom", "Trackbars", intialTracbarVals[3], hT, nothing)
 
def valTrackbars(wT=720, hT=540):
    widthTop = cv2.getTrackbarPos("Width Top", "Trackbars")
    heightTop = cv2.getTrackbarPos("Height Top", "Trackbars")
    widthBottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
    heightBottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")
    points = np.float32([(widthTop, heightTop), (wT-widthTop, heightTop),
                      (widthBottom , heightBottom ), (wT-widthBottom, heightBottom)])
    return points
 
def drawPoints(img,points):
    for x in range(4):
        cv2.circle(img,(int(points[x][0]),int(points[x][1])),15,(0,0,255),cv2.FILLED)
    return img
 
def getHistogram(img,minPer=0.1,display= False,region=1):
 
    if region ==1:
        histValues = np.sum(img, axis=0)
    else:
        histValues = np.sum(img[img.shape[0]//region:,:], axis=0)
 
    #print(histValues)
    maxValue = np.max(histValues)
    minValue = minPer*maxValue
 
    indexArray = np.where(histValues >= minValue)
    basePoint = int(np.average(indexArray))
    #print(basePoint)
 
    if display:
        imgHist = np.zeros((img.shape[0],img.shape[1],3),np.uint8)
        for x,intensity in enumerate(histValues):
            cv2.line(imgHist,(x,img.shape[0]),(x,img.shape[0]-intensity//255//region),(255,0,255),1)
            cv2.circle(imgHist,(basePoint,img.shape[0]),20,(0,255,255),cv2.FILLED)
        return basePoint,imgHist
 
    return basePoint
 
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

frameCounter = 0
intialTrackBarVals = [193,196,104,469]
initializeTrackbars(intialTrackBarVals)

init_val = [82,130,144,255,6,255]

cv2.namedWindow("HSV")
cv2.resizeWindow("HSV", 540, 240)
cv2.createTrackbar("Hue Min", "HSV", init_val[0],179, nothing)
cv2.createTrackbar("Hue Max", "HSV", init_val[1],179, nothing)
cv2.createTrackbar("Sat Min", "HSV", init_val[2],255, nothing)
cv2.createTrackbar("Sat Max", "HSV", init_val[3],255, nothing)
cv2.createTrackbar("Val Min", "HSV", init_val[4],255, nothing)
cv2.createTrackbar("Val Max", "HSV", init_val[5],255, nothing)

while True:
    # frameCounter += 1
    # if cap.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    #     frameCounter = 0
    success, img = cap.read()
    img = cv2.resize(img,(720,540))
    imgCopy = img.copy()
    imgResult = img.copy()

    #### Threshold
    imgHsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("Hue Min", "HSV")
    h_max = cv2.getTrackbarPos("Hue Max", "HSV")
    s_min = cv2.getTrackbarPos("Sat Min", "HSV")
    s_max = cv2.getTrackbarPos("Sat Max", "HSV")
    v_min = cv2.getTrackbarPos("Val Min", "HSV")
    v_max = cv2.getTrackbarPos("Val Max", "HSV")

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHsv, lower, upper)
    imgMask = cv2.bitwise_and(img, img, mask=mask)

    #### Warp Image
    hT, wT, c = img.shape
    points = valTrackbars()
    imgWarp = warpImg(mask,points,wT,hT)
    imgWarpPoints = drawPoints(imgMask,points)
    
    #### Finding Curve
    middlePoint,imgHist = getHistogram(imgWarp,display=True,minPer=0.5,region=4)
    curveAveragePoint, imgHist = getHistogram(imgWarp, display=True, minPer=0.7)
    curveRaw = curveAveragePoint - middlePoint
 
    #### Optimize Curve
    curveList.append(curveRaw)
    if len(curveList)>avgVal:
        curveList.pop(0)
    curve = int(sum(curveList)/len(curveList))
 
    #### Display
    height = imgWarp.shape[0]
    width = imgWarp.shape[1]
    region_of_interest_vertices = [
        (0, height),
        (width/2, height/2),
        (width, height)
    ]
    imgInvWarp = warpImg(imgWarp, points, wT, hT, inv=True)
    imgInvWarp = cv2.cvtColor(imgInvWarp, cv2.COLOR_GRAY2BGR)
    gray_image = cv2.cvtColor(imgInvWarp, cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(gray_image, 100, 120)
    cropped_image = region_of_interest(canny_image,
                    np.array([region_of_interest_vertices], np.int32),)
    lines = cv2.HoughLinesP(canny_image,
                            rho=2,
                            theta=np.pi/180,
                            threshold=50,
                            lines=np.array([]),
                            minLineLength=40,
                            maxLineGap=100)
    if lines is None:
        imgResult = img
    else:
        imgResult = draw_the_lines(img, lines)

    midY = 450
    cv2.putText(imgResult, "curve : " + str(curve), (20, 85), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
    cv2.line(imgResult, (wT // 2, midY), (wT // 2 + (curve * 3), midY), (255, 0, 255), 5)
    cv2.line(imgResult, ((wT // 2 + (curve * 3)), midY - 25), (wT // 2 + (curve * 3), midY + 25), (0, 255, 0), 5)

    ## FPS
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(imgResult, "FPS : " + str(int(fps)),(20,40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    for x in range(-30, 30):
        w = wT // 20
        cv2.line(imgResult, (w * x + int(curve // 50), midY - 10),
                 (w * x + int(curve // 50), midY + 10), (0, 0, 255), 2)
 
    #### NORMALIZATION
    curve = curve/100
    if curve>1: curve ==1
    if curve<-1:curve == -1

    print(curve)
    imgStacked = stackImages(0.5, ([img, imgWarpPoints, imgWarp],
                                             [imgHist, imgInvWarp, imgResult]))
    cv2.imshow('ImageStack', imgStacked)
    # cv2.imshow('Vid',imgResult)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break