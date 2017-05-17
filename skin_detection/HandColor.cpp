
#include<iostream>
#include <opencv2/objdetect/objdetect.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
//#include<algorithm>

using namespace std;
using namespace cv;

struct BGR {
	uchar b;
	uchar g;
	uchar r;
}
;;

int  handColor(Mat &frame1, VideoCapture &cap1) {
	/** This function is intended to store the color of hand within the rectangle of interest */
	int p;
	Mat frame2;
	Mat preCrop;
	cout << "\n Press Space when ready to caputure hand color...\n";
	Rect handR = Rect(100, 200, 100, 100); //Size of Rectangle
	for (;;) { //Archetypal Capture event
		cap1 >> frame2;
		rectangle(frame2, handR, Scalar(0, 0, 255), 1);
		imshow("Place center of hand here...", frame2);
		if (cvWaitKey(1) == 32) {
			cap1.retrieve(preCrop);
			break;
		};
	}
	cvDestroyWindow("Place center of hand here...");
	rectangle(preCrop, handR, Scalar(0, 0, 255), 1);
	imshow("Test", preCrop);
	Mat croppedImage = preCrop(handR);
	Mat postCrop;
	resize(croppedImage, croppedImage, Size(100,100));
	
	Rect collection = Rect(5, 5, 92, 92);
	//This Line tests rectangle placement. Ignore. rectangle(croppedImage, collection, Scalar(255, 0, 0), 1);
	imshow("Collected Hand", croppedImage);
	vector<BGR> colors;
	vector<double> colorsAvg;
	for (int y = 0; y < croppedImage.rows; y++)
	{
		for (int x = 0; x < croppedImage.cols; x++)
		{
			// get pixel
			colors.push_back(croppedImage.ptr<BGR>(y)[x]);
		}
	}
			//Determine average color:
	double colorsSum = 0;
	for (int i = 0; i < colors.size(); i++) {
		colorsAvg.push_back(((colors[i].b + colors[i].g + colors[i].r) / 3));
	}
	for (int i = 0; i < colors.size(); i++) {
		colorsSum += colorsAvg[i];
	};
	colorsSum /= colors.size();
	while (1) {
		if (cvWaitKey(1) == 32) {
			cvDestroyWindow("Test");
			cvDestroyWindow("Collected Hand");
			break;
		}
	}


	return colorsSum;
	}

		;

void findSkin(VideoCapture &cap1, int thresher){
	Mat frame3;
	int lo = 20;
	int hi = 20;
	int bins = 25;
	namedWindow("Live Feed", WINDOW_AUTOSIZE);
	createTrackbar("Low thresh", "Live feed", &lo, 255, 0);
	createTrackbar("High thresh", "Live feed", &hi, 255, 0);
	while (1) {
		cap1 >> frame3;
		Mat hsv; Mat hue;
		cvtColor(frame3, hsv, CV_BGR2HSV);
		imshow("Live Feed", frame3);


		MatND hist;
		int h_bins = 30; int s_bins = 32;
		int histSize[] = { h_bins, s_bins };

		float h_range[] = { 0, 179 };
		float s_range[] = { 0, 255 };
		const float* ranges[] = { h_range, s_range };

		int channels[] = { 0, 1 };

		/// Get the Histogram and normalize it
		calcHist(&hsv, 1, channels, Mat(), hist, 2, histSize, ranges, true, false);

		normalize(hist, hist, 0, 255, NORM_MINMAX, -1, Mat());

		/// Get Backprojection
		MatND backproj;
		calcBackProject(&hsv, 1, channels, hist, backproj, ranges, 1, true);
		Size a(5, 5);
		Mat disc;
		disc = getStructuringElement(MORPH_ELLIPSE, a);
		filter2D(backproj, disc, -1, backproj);
		Mat finalOut1;
		threshold(backproj, finalOut1, thresher, 255, 0);
		erode(finalOut1, finalOut1, Mat(), Point(-1, -1), 2);
		dilate(finalOut1, finalOut1, Mat(), Point(-1, -1), 2);
		const Mat *n3;
		n3 = new Mat[3];
		/**finalOut1.copyTo(n3[0]);
		finalOut1.copyTo(n3[1]);
		finalOut1.copyTo(n3[2]); */
		//merge(n3, 3, finalOut1);
		/// Draw the backproj
		imshow("BackProj", finalOut1);
		if (cvWaitKey(1) == 32) {
			break;
		}
	}

	}



int main() {
	bool x=1;
	VideoCapture cap(0);
	char k;
	//set up
	cvStartWindowThread(); //Prepares for later Window Destruction
	Mat frame;
	for (;;){
		
		cap >> frame; // get a new frame from camera        
		imshow("Video", frame);
		if (cvWaitKey(30) == 32) {
			cvDestroyWindow("Video");
			break;
		}
	}
		int j = handColor(frame,cap);
		findSkin(cap, j);

return 0;
}
