#include<iostream>
#include <opencv2/objdetect/objdetect.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
//#include<algorithm>

using namespace std;
using namespace cv;

int threshold_val = 127;

struct Hand_coordinates {
	int x_co;
	int y_co;
};
/*	Setup function.
	- creates window and trackbar to adjust threshold
*/
void setup_window(CascadeClassifier &face_cascade) {
	namedWindow("live feed", CV_WINDOW_AUTOSIZE);
	createTrackbar("Set Threshold Value", "live feed", &threshold_val, 255);
	face_cascade.load("C:/Users/Magnus/Documents/opencv/build/etc/haarcascades/haarcascade_frontalface_alt.xml");
	return;
}
/*-	Draw palm function
	- define a rectangular RegionOfInterest (roi)
	- apply Gaussian Blur to remove noise, perform thresholding
	- Finds center of contrast figure (palm)
	- draws roi window
*/
Hand_coordinates draw_palm_roi(Mat &frame) {
	Rect roi_ = Rect(50, 50, 250, 250); //Size of Rectangle
	Mat roi = frame(roi_); //get frame from video stream
	cvtColor(roi, roi, CV_BGR2GRAY); //convert to grayscale
	GaussianBlur(roi, roi, Size(5, 5), 5, 5); //Apply smoothing function - accuracy
	threshold(roi, roi, threshold_val, 255, CV_THRESH_BINARY_INV); //Convert feed into contrast
	//Mat se = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	//dilate(roi, roi, se); //Functions that smooth the image ... dunno if we need

	 /*-calc centroid of the palm */ //-Magical
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
	Hand_coordinates palm = { cx, cy };
	line(frame, Point(cx + roi_.x, cy + roi_.y), Point(cx + roi_.x, cy + roi_.y), Scalar(0, 0, 255), 5); //centroid
	rectangle(frame, roi_, Scalar(255, 0, 0), 2); //draw a rectangle on frame

	imshow("roi", roi);
	return palm;

}

/* Facedetection function
	- collect faces with Haar Cascade analysis.
*/
vector<Mat> facedetection(Mat &frame, CascadeClassifier &face_cascade) {
	Mat gray; //Frame where the faces will reside
	vector<Rect> faces; //collection of rectangle sizes
	vector<Mat> faces_comp; //collection of faces
	face_cascade.detectMultiScale(frame, faces, 1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_SCALE_IMAGE, Size(100, 100)); //For better performance set min face size to 100x100
	for (int i = 0; i < faces.size(); i++) {

		Point pt1(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
		Point pt2(faces[i].x, faces[i].y);
		rectangle(frame, pt1, pt2, cvScalar(0, 255, 0, 0), 1, 0, 0); //Rectangle around the face
		//Mat face = gray(faces[i]); //convert the position of the face into the face. 
		//resize(face, face, Size(50, 50), 1.0, 1.0, INTER_CUBIC);
		/*
		cv::Mat skin;
		//first convert our RGB image to YCrCb
		cvtColor(frame, skin, cv::COLOR_BGR2YCrCb);
		//uncomment the following line to see the image in YCrCb Color Space
		imshow("YCrCb Color Space", skin);
		//filter the image in YCrCb color space
		inRange(skin, cv::Scalar(0, 133, 77), cv::Scalar(255, 173, 127), skin);
		
		//findContours(skin, skin, 1, 1 );
		imshow("WOw", skin);
		*/
		Rect roi_ = Rect(50, 50, 250, 250);
		Rect face = Rect(pt1.x-faces[i].width, pt1.y-faces[i].height, faces[i].width, faces[i].height);
		Mat roi = frame(face);
		resize(roi, roi, Size(50, 50));
		imshow("riu1," +i , roi);
		faces_comp.push_back(roi);

	}
	//imshow("test", face);
	
	return faces_comp;
}

void start_capture(VideoCapture &cap, Mat &frame, CascadeClassifier &face_cascade) {
	while (true) {
		cap >> frame;
		Hand_coordinates palm = draw_palm_roi(frame); //draw rectangle and central point of palm.
		vector<Mat> faces = facedetection(frame, face_cascade);
		cout << faces.size() << endl;
		imshow("live feed", frame);
		for (int i = 0; i < faces.size(); ++i) {
			cout << faces[i].size() << endl;
		}
		cvWaitKey(10);
	}
	return;
}

int main() {
	cout << "Hello! Welcome to STO-detect" << endl;
	//set up
	VideoCapture cap(0);
	Mat frame;
	cap >> frame; //initial zero frame
	CascadeClassifier face_cascade;	//create cascade classifier used for the face detection
	setup_window(face_cascade);



	// start capture
	start_capture(cap, frame, face_cascade);

	return 0;
}


/* Functions to use in the future
	
	putText(frame, result, Point(frame.cols - 100, 80), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 4); //Write text into frame


*/