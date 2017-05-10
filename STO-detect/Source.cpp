#include<iostream>
#include <opencv2\objdetect\objdetect.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<algorithm>

using namespace std;
using namespace cv;

int threshold_val = 127;

int main() {
	VideoCapture cap(0);
	Mat frame;
	cap >> frame; //initial zero frame
	namedWindow("live feed", CV_WINDOW_AUTOSIZE);
	createTrackbar("Set Threshold Value", "live feed", &threshold_val, 255);

	//create cascade classifier used for the face detection
	CascadeClassifier face_cascade;
	//use the haarcascade_frontalface_alt.xml library
	face_cascade.load("C:/Users/Magnus/Documents/opencv/build/etc/haarcascades/haarcascade_frontalface_alt.xml");

	while (true) {
		cap >> frame;

		/*-define a rectangular RegionOfInterest (roi)
		-apply Gaussian Blur to remove noise
		-perform thresholding to segment palm (define a trackbar to adjust the threshold)
		-dilate to remove noise
		*/
		Rect roi_ = Rect(50, 50, 250, 250); //Size of Rectangle
		Mat roi = frame(roi_); //get frame from video stream
		cvtColor(roi, roi, CV_BGR2GRAY); //convert to grayscale
		GaussianBlur(roi, roi, Size(5, 5), 5, 5); //Apply smoothing function - accuracy
		threshold(roi, roi, threshold_val, 255, CV_THRESH_BINARY_INV); //Convert feed into contrast
		//Mat se = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
		//dilate(roi, roi, se); //Functions that smooth the image ... dunno if we need

		/*-calc centroid of the palm
		magical algorithm that I have no idea how it works
		*/
		float cx = 0.0, cy = 0.0; //starting values
		float sumi = 1.0;
		for (int i = 0; i < roi.rows; i++) {
			for (int j = 0; j<roi.cols; j++) {
				cy += roi.at<uchar>(i, j) * i; 
				cx += roi.at<uchar>(i, j) * j;
				sumi += roi.at<uchar>(i, j);
			}
		}
		cx /= sumi;
		cy /= sumi;
		
		/* collect faces to send to facerecognition
			- Haar Cascade analysis.
			- For better performance set min face size to 100x100
		*/
		vector<Rect> faces;
		face_cascade.detectMultiScale(frame, faces, 1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_SCALE_IMAGE, Size(100, 100));
		for (int i = 0; i < faces.size(); i++) {
			Point pt1(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
			Point pt2(faces[i].x, faces[i].y);

			rectangle(frame, pt1, pt2, cvScalar(0, 255, 0, 0), 1, 0, 0);
		}

		/* Draw everything!
			- line is the point of center of palm
			- rectangle is a fixed object
			- frame now contains {line,rectangle, vector of faces}
		*/
		line(frame, Point(cx + roi_.x, cy + roi_.y), Point(cx + roi_.x, cy + roi_.y), Scalar(0, 0, 255), 5); //centroid
		rectangle(frame, roi_, Scalar(255, 0, 0), 2); //draw a rectangle on frame
		imshow("live feed", frame);
		imshow("roi", roi);
		cvWaitKey(10);

	}
	cvWaitKey(0);
	return 0;
}


/* Functions to use in the future
	
	putText(frame, result, Point(frame.cols - 100, 80), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 4); //Write text into frame


*/