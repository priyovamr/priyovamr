from random import gauss
import cv2
import numpy as np
import time

video = cv2.VideoCapture('C:/Users/User/OneDrive - mail.ugm.ac.id/Dokumen/Github_ku/priyovamr/project_video.mp4')
# video = cv2.VideoCapture(0)

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

def warpImg(img,points,w,h,inv = False):
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    if inv:
        matrix = cv2.getPerspectiveTransform(pts2, pts1)
    else:
        matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgWarp = cv2.warpPerspective(img,matrix,(w,h))
    return imgWarp

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
            # cv2.line(imgHist,(x,img.shape[0]),(x,img.shape[0]-intensity//255//region),(255,0,255),1)
            cv2.circle(imgHist,(basePoint,img.shape[0]),20,(0,255,255),cv2.FILLED)
        return basePoint,imgHist
 
    return basePoint

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

def threshold_rel(img, lo, hi):
    vmin = np.min(img)
    vmax = np.max(img)
    
    vlo = vmin + (vmax - vmin) * lo
    vhi = vmin + (vmax - vmin) * hi
    return np.uint8((img >= vlo) & (img <= vhi)) * 255

def threshold_abs(img, lo, hi):
    return np.uint8((img >= lo) & (img <= hi)) * 255

def shadow_remove(img):
    rgb_planes = cv2.split(img)
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_norm_planes.append(norm_img)
    shadowremov = cv2.merge(result_norm_planes)
    return shadowremov

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

intialTrackBarVals = [316,329,132,462]
initializeTrackbars(intialTrackBarVals)

 # Arrays to store object points and image points from all the images
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane
curveList = []
result_planes = []
result_norm_planes = []
avgVal=10
pTime = 0
cTime = 0

while True :
    ret, imageN = video.read()
    image = cv2.resize(imageN, (720, 540))
    imgCopy = image.copy()
    # img2 = image.copy()
    hT, wT, c = image.shape
    points = valTrackbars()
    imgWarp = warpImg(image,points,wT,hT)
    gray = cv2.cvtColor(imgWarp, cv2.COLOR_BGR2GRAY)

    ## Laplacian
    # imgGauss = cv2.GaussianBlur(gray,(3,3),0)
    # img2 = cv2.Laplacian(imgGauss,cv2.CV_64F)

    # Threshold_abs
    hls = cv2.cvtColor(imgWarp, cv2.COLOR_RGB2HLS)
    hsv = cv2.cvtColor(imgWarp, cv2.COLOR_RGB2HSV)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    v_channel = hsv[:,:,2]

    right_lane = threshold_rel(l_channel, 0.5, 1.0)

    left_lane = threshold_abs(h_channel, 20, 30)
    left_lane &= threshold_rel(v_channel, 0.7, 1.0)
    left_lane[:,550:] = 0

    img2 = left_lane | right_lane

    ## Sobel xy
    # gray = cv2.cvtColor(imgWarp, cv2.COLOR_BGR2GRAY)
    # gauss = cv2.GaussianBlur(gray,(3,3),0)
    # grad_xy = cv2.Sobel(gauss, cv2.CV_64F, 1, 1, ksize=3)
    # img2 = grad_xy

    ## sobel
    # ret,binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    # imgGauss = cv2.GaussianBlur(binary,(3,3),0)

    # grad_x = cv2.Sobel(binary, cv2.CV_8U, 1, 0, ksize=3)
    # grad_y = cv2.Sobel(binary, cv2.CV_8U, 0, 1, ksize=3)

    # abs_grad_x = cv2.convertScaleAbs(grad_x)
    # abs_grad_y = cv2.convertScaleAbs(grad_y)

    # img_both = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    # kernel = np.ones((3, 3))
    # img2 = cv2.dilate(img_both, kernel, iterations=2)

    ### Optimize Warping
    imgWarpPoints = drawPoints(imgCopy,points)
    imgInvWarp = warpImg(img2, points, wT, hT, inv=True)

    #### Finding Curve
    middlePoint,imgHist = getHistogram(img2,display=True,minPer=0.3,region=3)
    curveAveragePoint, imgHist = getHistogram(img2, display=True, minPer=0.7)
    curveRaw = curveAveragePoint - middlePoint
    
    #### Optimize Curve
    curveList.append(curveRaw)
    if len(curveList)>avgVal:
        curveList.pop(0)
    curve = int(sum(curveList)/len(curveList))

    lines = cv2.HoughLinesP(imgInvWarp,
                            rho=2,
                            theta=np.pi/180,
                            threshold=50,
                            lines=np.array([]),
                            minLineLength=40,
                            maxLineGap=100)
    if lines is None:
        imgResult = image
    else:
        imgResult = draw_the_lines(image, lines)
    
    midY = 450
    cv2.putText(imgResult, "curve : " + str(curve), (20, 85), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
    cv2.line(imgResult, (wT // 2, midY), (wT // 2 + (curve * 3), midY), (255, 0, 255), 5)
    cv2.line(imgResult, ((wT // 2 + (curve * 3)), midY - 25), (wT // 2 + (curve * 3), midY + 25), (0, 255, 0), 5)

    for x in range(-30, 30):
        w = wT // 20
        cv2.line(imgResult, (w * x + int(curve // 50), midY - 10),
                 (w * x + int(curve // 50), midY + 10), (0, 0, 255), 2)
 
    #### NORMALIZATION
    curve = curve/100
    if curve>1: curve ==1
    if curve<-1:curve == -1

    print(curve)
    
     ## FPS
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(imgResult, "FPS : " + str(int(fps)),(20,40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    imgStacked = stackImages(0.7, ([image, imgWarpPoints],
                                    [img2, imgResult]))
    cv2.imshow('ImageStack', imgStacked)
    # cv2.imshow('Hasil', imgResult)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break