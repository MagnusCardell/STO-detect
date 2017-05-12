import cv2
import imutils
import argparse
import numpy as np

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())
 
# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
	camera = cv2.VideoCapture(0)
	time.sleep(0.25)
 
# otherwise, we are reading from a video file
else:
	camera = cv2.VideoCapture(args["video"])

#camera = cv2.VideoCapture('coaster.mp4')
#camera = cv2.VideoCapture('stretching.mp4')
camera = cv2.VideoCapture('tv.mp4')

roi = cv2.imread('hand2.jpg')
hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)


#target = cv2.imread('family3.jpg')
#hsvt = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)
while(camera.isOpened()):

    ret, target = camera.read()

    r = 800.0 / target.shape[1]
    dim = (800, int(target.shape[0] * r))
 
    target = cv2.resize(target, dim, interpolation = cv2.INTER_AREA)

    hsvt = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)

    # calculating object histogram
    roihist = cv2.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
    # normalize histogram and apply backprojection
    cv2.normalize(roihist,roihist,0,255,cv2.NORM_MINMAX)
    dst = cv2.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],1)
    # Now convolute with circular disc
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    cv2.filter2D(dst,-1,disc,dst)
    # threshold and binary AND
    ret,thresh = cv2.threshold(dst,50,255,0)
#    thresh = cv2.erode(thresh, None, iterations=2)
#    thresh = cv2.dilate(thresh, None, iterations=2)

    cnts = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    if (len(cnts)> 1):
        c = max(cnts, key=cv2.contourArea)
        cv2.drawContours(target,[c],-1,(0,255,255), 2)
    
    thresh = cv2.merge((thresh,thresh,thresh))
    
    res = cv2.bitwise_and(target,thresh)
    #res = np.vstack((target,thresh,res))
    
    #cv2.imshow('frame',np.hstack([frame, res]))

    cv2.imshow('frame',target)
    cv2.imshow('result',res)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

#cv2.imshow('res.jpg',res)
#cv2.waitKey(0)
#cv2.imwrite('res.jpg',res)

