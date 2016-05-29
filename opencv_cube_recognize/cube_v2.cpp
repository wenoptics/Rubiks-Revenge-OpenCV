#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>


using namespace cv;
using namespace std;

int pCannyT1 = 24;
int pCannyT2 = 146;

/// Param: HoughLine
int pHoughT = 28;

/// Param: Adaptive Threshold
int pAdaptThres_blockSize = 26;
int pAdaptThres_C = 2;

/// Param: morph
int pMorph_size = 2;

void showImg();

static void on_trackbar(int, void*) {
	showImg();
}

void init() {

	namedWindow("Control", CV_WINDOW_AUTOSIZE);

	createTrackbar("pMorph_size", "Control", &pMorph_size, 20, on_trackbar);

	createTrackbar("pCannyT1", "Control", &pCannyT1, 1000, on_trackbar);
	createTrackbar("pCannyT2", "Control", &pCannyT2, 1000, on_trackbar);

}



vector<vector<Point> > contours;
Mat imgSource;

void showImg(){


	Mat imgGrad, imgCanny;
	cvtColor(imgSource, imgGrad, CV_BGR2GRAY);

	//Mat imgThresholded = imgGrad.clone();

	Mat imgMorph = Mat();

	Mat element = getStructuringElement(MorphShapes::MORPH_RECT,
		Size(2 * pMorph_size + 1, 2 * pMorph_size + 1), Point(pMorph_size, pMorph_size));

	/// do MORPH_Open	
	cv::morphologyEx(imgGrad, imgMorph, MORPH_OPEN, element);

	/// do MORPH_Close
	cv::morphologyEx(imgMorph, imgMorph, MORPH_CLOSE, element);

	imshow("morphological", imgMorph);


	Canny(imgMorph, imgCanny, pCannyT1, pCannyT2, 3);
	imshow("Canny", imgCanny);

	Mat imgCvtBin;// = imgGrad.clone();
	imgCanny.convertTo(imgCvtBin, CV_8U);

	//finding all contours in the image
	findContours(imgCvtBin, contours, CV_RETR_LIST, CHAIN_APPROX_SIMPLE);

	Mat result(imgCvtBin.size(), CV_8U, cv::Scalar(255));
	drawContours(result, contours, -1, cv::Scalar(0), 2);

	imshow("contours", result);


	Mat imgPolyResult(imgCvtBin.size(), imgSource.type());

	printf("we have contours.size()==%d \n", contours.size());

	//iterating through each contour
	for (int i = 0; i < contours.size(); i++) {
		vector<Point> poly;
		approxPolyDP(Mat(contours[i]), poly,
			5, // accuracy of the approximation  
			true); // yes it is a closed shape  


		int r = rand() % 255;
		int g = rand() % 255;
		int b = rand() % 255;

		if (poly.size() == 4) {
			polylines(imgPolyResult, poly, true, CV_RGB(r, g, b), 3);
		}
		else {
			polylines(imgPolyResult, poly, true, CV_RGB(r, r, r), 0.5);
		}

	}
	imshow("imgPolyResult", imgPolyResult);


}

int key;
int main() {
	init();

	imgSource = imread("D:\\WorkSpace\\WIN\\VS2013\\CSharp\\opencv_cube_recognize\\raw\\20160526_155743.jpg");

	resize(imgSource, imgSource, Size(0, 0), 0.2, 0.2);
	imshow("src img(resized)", imgSource);

	showImg();


	while (true){
		key = cvWaitKey(0);
		if (key == 27){
			break;
		}
	}

	return 0;

}
