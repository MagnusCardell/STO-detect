import cv2
import numpy as np

camera = cv2.VideoCapture('coaster.mp4')
#camera = cv2.VideoCapture('stretching.mp4')

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
    thresh = cv2.erode(thresh, None, iterations=4)
    thresh = cv2.dilate(thresh, None, iterations=4)
    
    thresh = cv2.merge((thresh,thresh,thresh))
    
    res = cv2.bitwise_and(target,thresh)
    #res = np.vstack((target,thresh,res))
    
    #cv2.imshow('frame',np.hstack([frame, res]))

    cv2.imshow('thresh',thresh)
    cv2.imshow('frame',target)
    cv2.imshow('result',res)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

#cv2.imshow('res.jpg',res)
#cv2.waitKey(0)
#cv2.imwrite('res.jpg',res)

