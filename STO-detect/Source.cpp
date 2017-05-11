#include<iostream>
#include <opencv2/objdetect/objdetect.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
//#include<algorithm>

using namespace std;
using namespace cv;

int threshold_val = 127;

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
vector<Mat> draw_palm_roi(Mat &frame) {
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

	line(frame, Point(cx + roi_.x, cy + roi_.y), Point(cx + roi_.x, cy + roi_.y), Scalar(0, 0, 255), 5); //centroid
	rectangle(frame, roi_, Scalar(255, 0, 0), 2); //draw a rectangle on frame
	imshow("roi", roi);
	vector<Mat> palm_frames; //future place for frames of palms
	return palm_frames;
}

/* Facedetection function
	- collect faces with Haar Cascade analysis.
*/
vector<Rect> facedetection(Mat &frame, CascadeClassifier &face_cascade) {
	Mat gray, face; //Frame where the faces will reside
	vector<Rect> faces; //collection of rectangle sizes
	face_cascade.detectMultiScale(frame, faces, 1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_SCALE_IMAGE, Size(100, 100)); //For better performance set min face size to 100x100
	for (int i = 0; i < faces.size(); i++) {
		Point pt1(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
		Point pt2(faces[i].x, faces[i].y);
		rectangle(frame, pt1, pt2, cvScalar(0, 255, 0, 0), 1, 0, 0); //Rectangle around the face
		//Mat face = gray(faces[i]); //convert the position of the face into the face. 
		//resize(face, face, Size(50, 50), 1.0, 1.0, INTER_CUBIC);

	}
	//imshow("test", face);
	
	return faces;
}

void start_capture(VideoCapture &cap, Mat &frame, CascadeClassifier &face_cascade) {
	while (true) {
		cap >> frame;
		vector <Mat> palms = draw_palm_roi(frame); //draw rectangle and central point of palm.
		vector<Rect> faces = facedetection(frame, face_cascade);

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