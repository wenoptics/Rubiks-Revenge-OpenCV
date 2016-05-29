#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include "alg.h"

#define useAdaptiveThresold
#undef useAdaptiveThresold
#undef useHoughLinePAtLast

using namespace cv;
using namespace std;

int iOE = 8;
int iOD = 3;
int iCE = 7;
int iCD = 3;

int pCannyT1 = 24;
int pCannyT2 = 146;

int pAdaptThres_blockSize = 26;
int pAdaptThres_C = 2;

int pHoughLineTr = 80, pHoughLineMinLineLen=129, pHoughLineMaxLineGap=48;

int pGaussianBlurSize = 1;

int pSortedGroupingGap = 150;

void showImg();

static void on_trackbar(int, void*) {
	showImg();
}

void init() {
	
	namedWindow("Control", CV_WINDOW_NORMAL);

#ifdef useAdaptiveThresold

	createTrackbar("pAdaptThres_blockSize", "Control", &pAdaptThres_blockSize, 300, on_trackbar);
	createTrackbar("pAdaptThres_C", "Control", &pAdaptThres_C, 500, on_trackbar);
#endif

	createTrackbar("mOpen erode", "Control", &iOE, 20, on_trackbar);
	createTrackbar("mOpen dilate", "Control", &iOD, 20, on_trackbar);
	createTrackbar("mClose erode", "Control", &iCE, 20, on_trackbar);
	createTrackbar("mClose dilate", "Control", &iCD, 20, on_trackbar);


	createTrackbar("pGaussianBlurSize", "Control", &pGaussianBlurSize, 30, on_trackbar);
	

	createTrackbar("pCannyT1", "Control", &pCannyT1, 1000, on_trackbar);
	createTrackbar("pCannyT2", "Control", &pCannyT2, 1000, on_trackbar);
	
	createTrackbar("pSortedGroupingGap", "Control", &pSortedGroupingGap, 1500, on_trackbar);
	
	createTrackbar("pHoughLineTr", "Control", &pHoughLineTr, 200, on_trackbar);
	createTrackbar("pHoughLineMinLineLen", "Control", &pHoughLineMinLineLen, 500, on_trackbar);
	createTrackbar("pHoughLineMaxLineGap", "Control", &pHoughLineMaxLineGap, 100, on_trackbar);

}



vector<vector<Point> > contours;
Mat imgSource;

void showImg(){


	Mat imgGrad, imgCanny;
	cvtColor(imgSource, imgGrad, CV_BGR2GRAY);

#ifdef useAdaptiveThresold
	Mat imgThresholded = imgGrad.clone();

	cv::adaptiveThreshold(imgGrad, imgThresholded, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY,
		pAdaptThres_blockSize * 2 + 1, pAdaptThres_C / 10);
	cv::imshow("adative threshold", imgThresholded);

	Mat imgMorph = imgThresholded.clone();
#else
	Mat imgMorph = imgGrad.clone();
#endif

	//morphological opening (remove small objects from the foreground)
	if (iOE != 0)
		erode(imgMorph, imgMorph, getStructuringElement(MORPH_ELLIPSE, Size(iOE, iOE)));
	if (iOD != 0)
		dilate(imgMorph, imgMorph, getStructuringElement(MORPH_ELLIPSE, Size(iOD, iOD)));

	//morphological closing (fill small holes in the foreground)
	if (iCD != 0)
		dilate(imgMorph, imgMorph, getStructuringElement(MORPH_ELLIPSE, Size(iCD, iCD)));
	if (iCE != 0)
		erode(imgMorph, imgMorph, getStructuringElement(MORPH_ELLIPSE, Size(iCE, iCE)));

	imshow("morphological", imgMorph);


	Canny(imgMorph, imgCanny, pCannyT1, pCannyT2, 3);
	imshow("Canny", imgCanny);

	

#if 0
	Mat imgBlur = imgCanny.clone();
	GaussianBlur(imgBlur, imgBlur, Size(pGaussianBlurSize * 2 + 1, pGaussianBlurSize * 2 + 1), 0, 0);
	imshow("imgBlur", imgBlur);
#endif

	Mat imgCvtBin;// = imgGrad.clone();
	imgCanny.convertTo(imgCvtBin, CV_8U);

	//finding all contours in the image
	contours.clear();
	findContours(imgCvtBin, contours, CV_RETR_LIST, CHAIN_APPROX_NONE);

	Mat imgDrawContours(imgCvtBin.size(), CV_8U, cv::Scalar(255));
	drawContours(imgDrawContours, contours, -1, cv::Scalar(0), 2);

	imshow("contours", imgDrawContours);


	Mat imgPolyResult(imgCvtBin.size(), imgSource.type());
	Mat imgPolyOf4(imgCvtBin.size(), CV_8UC3);

	printf("we have contours.size()==%d \n", contours.size());

	vector< c_a_pair > goodContoursPaired;

	//iterating through each contour
	for (int i = 0; i < contours.size(); i++) {
		vector<Point> poly;

		double epsilon = 0.05*arcLength(contours[i], true);

		approxPolyDP(Mat(contours[i]), poly,
			5, // accuracy of the approximation  
			true); // yes it is a closed shape  
		

		int r = rand() % 255;
		int g = rand() % 255;
		int b = rand() % 255;

		if (poly.size() == 4 && isContourConvex(poly)) {
			double area = contourArea(poly);
			if (area > 0) {

				c_a_pair paired(poly, area);
				goodContoursPaired.push_back(paired);

				polylines(imgPolyResult, poly, true, CV_RGB(r, g, b), 3);
				polylines(imgPolyOf4, poly, true, CV_RGB(255, 255, 255), 2);
			}
		}
		else {
			polylines(imgPolyResult, poly, true, CV_RGB(r, r, r), 0.5);
		}

	}
	imshow("imgPolyResult", imgPolyResult);	

	printf("goodContours.count==%d \n", goodContoursPaired.size());

	cvtColor(imgPolyOf4, imgPolyOf4, CV_RGB2GRAY);
	imshow("imgPolyOf4", imgPolyOf4);


	//vector<c_a_pair> pairedGroup = makeContourAreaPair(contours);
	printAreas(goodContoursPaired);
	vector<c_a_pair> sortedGroup = sortPairedGroupByArea(goodContoursPaired);
	printAreas(sortedGroup);

	vector<vector<c_a_pair>> groupingGroups = sortedGrouping_continous(sortedGroup, pSortedGroupingGap);
	printGroups(groupingGroups);
	

	// HoughLinesP on imgPolyResult
	Mat imgLineDetec(imgCvtBin.size(), CV_8UC3);
	

	Mat imgPolyGrouping(imgCvtBin.size(), imgSource.type());
	for each (vector<c_a_pair> oneGroup in groupingGroups)
	{
		//if (oneGroup.size() < 16)
		//	continue;

		int r = rand() % 255;
		int g = rand() % 255;
		int b = rand() % 255;

		putText(imgPolyGrouping, std::to_string(oneGroup[0].second), oneGroup[0].first[0],
			FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, CV_RGB(r, g, b));

#ifdef useHoughLinePAtLast
		// HoughLinesP detect from 
		Mat imgIntentGroupLine(imgCvtBin.size(), imgSource.type());
		cvtColor(imgIntentGroupLine, imgIntentGroupLine, CV_RGB2GRAY);
#endif


		// we collect all the point from this group, 
		//		to get the minArea rect later.
		vector<Point> groupPoints;

		// we use this tmpBImage to calc the total area of all pieces
		//		(due to overlapping issues, cannot simply add sum)
		Mat tmpBImage(imgCvtBin.size(), CV_8U, Scalar(0, 0, 0));
		for each (c_a_pair ele in oneGroup)
		{

			//calc the area by drawing every single pieces
			fillConvexPoly(tmpBImage, ele.first, CV_RGB(255, 255, 255));

#ifdef useHoughLinePAtLast
			// draw lines on imgIntentGroupLine, later we will use HoughLinesP to enhance it.
			polylines(imgIntentGroupLine, ele.first, true, CV_RGB(255,255,255), 2);
#endif

			// this is for debug preview
			fillConvexPoly(imgPolyGrouping, ele.first, CV_RGB(r, g, b));			

			//polylines(imgPolyGrouping, ele.first, true, CV_RGB(r, g, b), 4);

			//putText(imgPolyGrouping, std::to_string(ele.second), ele.first[0], 
			//	FONT_HERSHEY_SCRIPT_SIMPLEX, 0.6, CV_RGB(r, g, b));

			for each (Point po in ele.first)
			{
				groupPoints.push_back(po); 
			}
		}

		// find and draw the minAreaRect
		RotatedRect rRect =  minAreaRect(Mat(groupPoints));
		Point2f pointsRect[4];
		rRect.points(pointsRect);
		line(imgPolyGrouping, pointsRect[0], pointsRect[1], CV_RGB(r, g, b), 1);
		line(imgPolyGrouping, pointsRect[1], pointsRect[2], CV_RGB(r, g, b), 2);
		line(imgPolyGrouping, pointsRect[2], pointsRect[3], CV_RGB(r, g, b), 1);
		line(imgPolyGrouping, pointsRect[3], pointsRect[0], CV_RGB(r, g, b), 2);

		double minAreaArea = rRect.size.area();
		int piecesTotalArea = countNonZero(tmpBImage);
		double areaRatio = (double)piecesTotalArea / minAreaArea;

		printf("area of this groupRect %.2f, area of pieces %d, ratio=%.2f \n", minAreaArea, piecesTotalArea, areaRatio);

		putText(imgPolyGrouping, 
			std::to_string((int)minAreaArea) + "/" + std::to_string(piecesTotalArea),
			Point(pointsRect[0]),
			FONT_HERSHEY_SCRIPT_SIMPLEX, 0.4, CV_RGB(255,255,255));

		if (oneGroup.size() > 16 && areaRatio > 0.4) 
		{

#ifdef useHoughLinePAtLast
			vector<Vec4i> lines;
			HoughLinesP(imgIntentGroupLine, lines, 1, CV_PI / 180, pHoughLineTr, pHoughLineMinLineLen, pHoughLineMaxLineGap);

			for (size_t i = 0; i < lines.size(); i++)
			{
				Vec4i l = lines[i];
				line(imgLineDetec, Point(l[0], l[1]), Point(l[2], l[3]), CV_RGB(r, g, b), 1, CV_AA);
			}
			imshow("imgLineDetec", imgLineDetec);
#endif


		}


	}

	imshow("imgPolyGrouping", imgPolyGrouping);

	//imshow("imgIntentGroupLine", imgIntentGroupLine);
		


}



int key_1;
int main() {

	init();

	// test sortedGrouping() 
	//vector < int > vList { 5, 8, 8, 21, 25, 26, 28, 40, 43, 55 };
	//printGroups( sortedGrouping(vList, 2) );



#if 1

	//20160526_155743
	//20160526_155725
	// bad 20160526_155723
	imgSource = imread("D:\\WorkSpace\\WIN\\VS2013\\CSharp\\opencv_cube_recognize\\raw\\20160526_155725.jpg");

	resize(imgSource, imgSource, Size(0, 0), 0.2, 0.2);
	imshow("src img(resized)", imgSource);	

	showImg();
#else
	// init capture
	VideoCapture cap;
	cap.open(0);

	// create new window  to show image

	while (1) {
		cap >> imgSource;

		// print the image;
		imshow("window", imgSource);
		showImg();

		// delay 33ms
		waitKey(22);

	}
	
#endif


	while (true){
		key_1 = cvWaitKey(0);
		if (key_1 == 27){
			break;
		}
	}

	return 0;

}
