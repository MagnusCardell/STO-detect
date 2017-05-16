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

struct pca_data {
	Mat average;
	vector<Mat> top4vectors;
	Mat image;
	vector<Mat> eigenfacesvector;
};

static  Mat formatImagesForPCA(const vector<Mat> &data) {
	Mat dst(static_cast<int>(data.size()), data[0].rows*data[0].cols, CV_32F);
	for (unsigned int i = 0; i < data.size(); i++) {
		Mat image_row = data[i].clone().reshape(1, 1);
		Mat row_i = dst.row(i);
		image_row.convertTo(row_i, CV_32F);
	}
	return dst;
}


// Normalizes a given image into a value range between 0 and 255.
//	Given the address to the matrix.
Mat norm_0_255(const Mat& src) {
	// Create and return normalized image:
	Mat dst;
	switch (src.channels()) {
	case 1:
		cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
		break;
	case 3:
		cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
		break;
	default:
		src.copyTo(dst);
		break;
	}
	return dst;
}

vector<Mat> face_trainer(VideoCapture &cap, Mat &frame, CascadeClassifier &face_cascade) {
	vector<Rect> faces; //collection of rectangle sizes
	vector<Mat> faces_comp; //collection of faces
	bool finding_10_faces = true;
	while (finding_10_faces) {
		cap >> frame;
		imshow("live feed", frame);
		face_cascade.detectMultiScale(frame, faces, 1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_SCALE_IMAGE, Size(100, 100)); //For better performance set min face size to 100x100
		for (int i = 0; i < faces.size(); i++) {
			Point pt1(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
			Point pt2(faces[i].x, faces[i].y);
			rectangle(frame, pt1, pt2, cvScalar(0, 255, 0, 0), 1, 0, 0); //Rectangle around the face

			Rect face = Rect(pt1.x - faces[i].width, pt1.y - faces[i].height, faces[i].width, faces[i].height);
			Mat roi = frame(face);
			resize(roi, roi, Size(50, 50));
			faces_comp.push_back(roi);
			if (faces_comp.size() >= 10) {
				finding_10_faces = false;
			}
		}
	}
	/*
	cout << faces_comp.size() << endl;
	int i = 1;
	for (auto c : faces_comp) {
		imshow("riu"+to_string(i), c);
		++i;
	}
	cvWaitKey(0);
	*/
	return faces_comp;
}

pca_data face_processing(vector<Mat> train) {
	//read first image then replace all values to create a blueprint
	Mat color = train[0];
	Mat image;
	cvtColor(color, image, CV_BGR2GRAY);
	Mat average = Mat::zeros(image.size(), image.type());

	//empty 2D array for accumulating pixel values over 255 and computing average
	int **collect;
	collect = new int*[50];
	for (int x = 0; x < 50; ++x) {
		collect[x] = new int[50];
		for (int y = 0; y < 50; ++y) {
			collect[x][y] = 0;
		}
	}
	//Vector of Matrices will collect all images
	//vector<Mat> faces;
	//read the rest of the images to add them up pixel by pixel to the collect array
	int s = 0;
	for (int i = 0; i < 10; ++i) {
		//1. read in the file
		cvtColor(train[i], train[i], CV_BGR2GRAY);
		Mat origin = train[i];
		//2. collect total values
		for (int n = 0; n < 50; n++) {
			for (int k = 0; k < 50; k++) {
				//Target pixel value and store it in collect array to perform pixel operations above 255
				// Scalar type is a 3-channel data type. We just want the first channel
				Scalar intensity = image.at<uchar>(n, k);
				collect[n][k] += intensity.val[0];
			}
		}

		//store image in vector
					//faces.push_back(image);
		//3. input average back into one image to get the AVERAGE face
		for (int n = 0; n < 50; n++) {
			for (int k = 0; k < 50; k++) {
				//Scalar type is necessary to keep the 3-channel data type of a Matrix
				Scalar intensity2 = average.at<uchar>(n, k);
				intensity2.val[0] = (collect[n][k]) / 10;
				average.at<uchar>(n, k) = intensity2.val[0];
			}
		}
	}
	//subtract average faces from every face:
	for (int i = 0; i<10; i++) {
		train[i] -= average;
	}
	//Get the covariance matrix
	Mat combine = formatImagesForPCA(train); //function for pushing all images into one matrix
	Mat covariance, mean2;
	//OpenCV function for getting the covariance matrix. Flag specify rows or columns
	calcCovarMatrix(combine, covariance, mean2, COVAR_ROWS | cv::COVAR_NORMAL);
	//cout << "covriance = " << endl << " " << covariance << endl << endl;
	cout << covariance.size() << endl;


	//Get Egienvalues and eigenvectors
	Mat eigenval, eigenvect;
	PCA pca(combine, Mat(), PCA::DATA_AS_ROW, 10);
	eigenval = pca.eigenvalues;
	eigenvect = pca.eigenvectors;
	cout << "eigenval = " << endl << " " << eigenval << endl << endl;
	//cout << "eigenvect = " << endl << " " << eigenvect << endl << endl;
	cout << "Eigenval size: " << eigenval.size() << " Eigenvect size: " << eigenvect.size() << endl;


	//Get the top 4 vectors from top 4 eigenvalues
	Mat top4vectors;
	int e_vectnumb = 4;
	for (int i = 0; i < e_vectnumb; i++) {
		top4vectors.push_back(eigenvect.row(i));
	}

	//Multiply each eigenvector with each of the (face - average) matrix
	//Vector of images and vectors
	vector<Mat> eigenfacesimage;
	vector<Mat> eigenfacesvector;

	for (int n = 0; n < 10; n++) {
		for (int i = 0; i < 4; i++) {
			//Nth row of (face-average) x ith row of eigenvector by component multiplcication .mul()
			Mat eigenfacevector = combine.row(n).mul(top4vectors.row(i));
			eigenfacesimage.push_back(norm_0_255(eigenfacevector).reshape(1, train[0].rows));
			eigenfacesvector.push_back(eigenfacevector);

		}
	}
	cout << "The amount of Eigenfaces are: " << eigenfacesimage.size() << endl;


	pca_data package{ average, top4vectors, image, eigenfacesvector };

	return package;
}


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
	vector<Rect> faces; //collection of rectangle sizes
	vector<Mat> faces_comp; //collection of faces
	face_cascade.detectMultiScale(frame, faces, 1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_SCALE_IMAGE, Size(100, 100)); //For better performance set min face size to 100x100
	for (int i = 0; i < faces.size(); i++) {

		Point pt1(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
		Point pt2(faces[i].x, faces[i].y);
		rectangle(frame, pt1, pt2, cvScalar(0, 255, 0, 0), 1, 0, 0); //Rectangle around the face
		//Mat face = gray(faces[i]); //convert the position of the face into the face. 
		//resize(face, face, Size(50, 50), 1.0, 1.0, INTER_CUBIC);
		
		//Set RGB values. check in range
		//Do Morphological operations
		//Dilate before erode
		cv::Mat skin;
		//first convert our RGB image to YCrCb
		cvtColor(frame, skin, cv::COLOR_BGR2YCrCb);
		//uncomment the following line to see the image in YCrCb Color Space
		imshow("YCrCb Color Space", skin);
		//filter the image in YCrCb color space
		inRange(skin, cv::Scalar(0, 133, 77), cv::Scalar(255, 173, 127), skin);
		
		//findContours(skin, skin, 1, 1 );
		imshow("WOw", skin);

		

		Rect face = Rect(pt1.x-faces[i].width, pt1.y-faces[i].height, faces[i].width, faces[i].height);
		Mat roi = frame(face);
		resize(roi, roi, Size(50, 50));
		imshow("riu1," +i , roi);
		faces_comp.push_back(roi);

	}
	//imshow("test", face);
	
	return faces_comp;
}

void calc_euclidian(vector<Mat> new_faces, pca_data &pca_result) {

	for (auto testimage : new_faces) {

		//Get the feature vector by subtracting the average of test phase
		testimage -= pca_result.average;
		imwrite("featurevector.jpg", testimage);
		//Convert the image to vector row project the image on the eigenspac
		Mat testvect2 = testimage.reshape(0, 1);
		Mat testvect;
		testvect2.convertTo(testvect, CV_32FC1);

		//Multiplication by components
		//Vector of images and vectors
		vector<Mat> imageprojection;
		vector<Mat> vectprojection;

		for (int i = 0; i < 4; i++) {
			//Nth row of (face-average) x ith row of eigenvector by component multiplcication .mul()
			//Mat temporary = testvect.mul(pca_result.top4vectors.row(i));
			//imageprojection.push_back(norm_0_255(temporary).reshape(1, pca_result.image.rows));
			//imwrite("testprojection" + to_string(i) + ".jpg", imageprojection.back());
			//vectprojection.push_back(temporary);
		}
		//Calculate Euclidian distance
		vector<float>euclidiandist;
		for (int i = 0; i < 10; i++) {
			for (int n = 0; n < 4; n++) {
				double dist = norm(pca_result.eigenfacesvector[i] - vectprojection[n], NORM_L2); //Euclidian distance
				euclidiandist.push_back(dist);
			}
		}

		float sum = 0;
		int threshhold = 30;
		bool face = 0;
		for (int i = 0; i < euclidiandist.size(); i++) {
			if (threshhold > euclidiandist[i]) {
				face = 1;
			}
			cout << euclidiandist[i] << endl;
			sum += euclidiandist[i];
		}
		float averagenumb = sum / euclidiandist.size();
		cout << "Average distance: " << averagenumb << endl;

		if (face) {
			cout << "This is a face" << endl;
		}
	}

}


void start_capture(VideoCapture &cap, Mat &frame, CascadeClassifier &face_cascade, pca_data &pca_training) {
	while (true) {
		cap >> frame;
		Hand_coordinates palm = draw_palm_roi(frame); //draw rectangle and central point of palm.
		vector<Mat> faces = facedetection(frame, face_cascade);
		cout << faces.size() << endl;

		imshow("live feed", frame);
		for (int i = 0; i < faces.size(); ++i) {
			cout << faces[i].size() << endl;
		}
		//calc_euclidian(faces, pca_training);
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

	vector<Mat> train_faces = face_trainer(cap, frame, face_cascade);
	pca_data result;
	//result = face_processing(train_faces);
	// start capture
	start_capture(cap, frame, face_cascade, result);

	return 0;
}


/* Functions to use in the future
	
	putText(frame, result, Point(frame.cols - 100, 80), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 4); //Write text into frame


*/
