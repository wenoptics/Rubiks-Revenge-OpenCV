#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include "alg.h"

#define useAdaptiveThresold0
#undef useAdaptiveThresold1
#undef useHoughLinePAtLast

using namespace cv;
using namespace std;

int iOE = 12; // 8;
int iOD = 11; // 3;
int iCE = 9; // 7;
int iCD = 1; //3;

int iLowH = 0;
int iHighH = 179;

int iLowS = 0;
int iHighS = 255;

int iLowV = 0;
int iHighV = 255;

int pCannyT1 = 167;
int pCannyT2 = 340;

int pDistThreshold = 19;

int pAdaptThres_blockSize = 7, pAdaptThres_C = 94;

int pHoughLineTr = 80, pHoughLineMinLineLen=129, pHoughLineMaxLineGap=48;

int pGaussianBlurSize = 1;

int pSortedGroupingGap = 30;

const int cTransformedSize = 80;

void showImg();

static void on_trackbar(int, void*) {
	showImg();
}

void init() {

	namedWindow("hsv", CV_WINDOW_AUTOSIZE);
	createTrackbar("LowH", "hsv", &iLowH, 179); //Hue (0 - 179)
	createTrackbar("HighH", "hsv", &iHighH, 179);

	createTrackbar("LowS", "hsv", &iLowS, 255); //Saturation (0 - 255)
	createTrackbar("HighS", "hsv", &iHighS, 255);

	createTrackbar("LowV", "hsv", &iLowV, 255); //Value (0 - 255)
	createTrackbar("HighV", "hsv", &iHighV, 255);
	
	namedWindow("Control", CV_WINDOW_NORMAL);

#ifdef useAdaptiveThresold0

	createTrackbar("pAdaptThres_blockSize", "Control", &pAdaptThres_blockSize, 300, on_trackbar);
	createTrackbar("pAdaptThres_C", "Control", &pAdaptThres_C, 500, on_trackbar);
#endif


	createTrackbar("pDistThreshold", "Control", &pDistThreshold, 100, on_trackbar);

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


vector<Point2f> VertexPersp;
void do_perspective_transform(Mat src, Mat& dst) {
	Mat matPTransform;

	// 0,0  0,h  w,h  w,0
	// w == 80 ; h == 80
	int w = cTransformedSize;
	int h = cTransformedSize;
	vector<Point2f> dst_transform
		= { Point2f(0, 0), Point2f(0, h), Point2f(w, h), Point2f(w, 0) };

	Mat imgAfterPerspTrans;

	matPTransform = cv::getPerspectiveTransform(&VertexPersp[0], &dst_transform[0]);

	cv::warpPerspective(src, imgAfterPerspTrans, matPTransform, Size(w, h));
	//imshow("AfterPerspTrans", imgAfterPerspTrans);

	dst = imgAfterPerspTrans;
}

void do_back_perspective_transform(Mat imgToTransform, Mat& dst_image, Size finalSize) {
	Mat matPTransform;

	// 0,0  0,h  w,h  w,0
	vector<Point2f> dst_transform
		= {
		Point2f(0, 0),
		Point2f(0, imgToTransform.rows),
		Point2f(imgToTransform.cols, imgToTransform.rows),
		Point2f(imgToTransform.cols, 0)
	};

	vector<Point2f> vec_4points = VertexPersp;

	if (vec_4points.size() == 0){
		vec_4points = dst_transform;
	}

	matPTransform = cv::getPerspectiveTransform(&dst_transform[0], &vec_4points[0]);
	cv::warpPerspective(imgToTransform, dst_image, matPTransform, finalSize);

}


void getBackPerspectiveTransformMatrix(Matx33f& dst, Mat sourceMat) {
	// 0,0  0,h  w,h  w,0
	vector<Point2f> dst_transform
		= {
		Point2f(0, 0),
		Point2f(0, sourceMat.rows),
		Point2f(sourceMat.cols, sourceMat.rows),
		Point2f(sourceMat.cols, 0)
	};

	vector<Point2f> vec_4points = VertexPersp;

	if (vec_4points.size() == 0){
		vec_4points = dst_transform;
	}

	Mat m = cv::getPerspectiveTransform(&dst_transform[0], &vec_4points[0]);
	dst = m;
}

Mat rot_mat;
Mat dstWarpAffine;
void get4x4PointsDetectionMat(Mat& dst, const Point& startPoint, double rotationAngle, int gap) {

	dst = Mat::zeros(dst.size(), CV_8U);
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			int row = i * gap + startPoint.y;
			int col = j * gap + startPoint.x;
			dst.at<uchar>(row, col) = 255;
		}
	}
	rot_mat = getRotationMatrix2D(startPoint, rotationAngle, 1.0);
	dstWarpAffine = Mat(dst.size(), CV_8U);// = dst.clone();
	warpAffine(dst, dstWarpAffine, rot_mat, dstWarpAffine.size());
	dst = Mat(dstWarpAffine);

	dstWarpAffine.release();
	rot_mat.release();
}

void get4x4PointsSet(vector<Point>& dst) {
	dst = vector<Point>();

	const int _full = cTransformedSize / 4;
	const int _half = _full / 2;
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			int row = i * _full + _half;
			int col = j * _full + _half;
			dst.push_back(Point(i, j));
		}
	}
}


vector<vector<Point> > contours;
Mat imgSource;

void showImg(){


	Mat imgGrad, imgCanny, imgHSV;
	/*

	cvtColor(imgSource, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV

	Mat imgThresholded;

	inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image

	imshow("imgHSV", imgThresholded);

	return;
	*/

	cvtColor(imgSource, imgGrad, CV_HSV2BGR);
	cvtColor(imgGrad, imgGrad, CV_BGR2GRAY);
	
	//equalizeHist(imgGrad, imgGrad);	

	Canny(imgGrad, imgCanny, pCannyT1, pCannyT2, 3);
	imshow("Canny", imgCanny);

	// inverse the canny output, (make edges to black lines)
	bitwise_not(imgCanny, imgCanny);
	//threshold(imgCanny, imgCanny, 1, 255, CV_THRESH_BINARY_INV);
	imshow("Inversed Canny", imgCanny);

	// cover the black edges on the origin image
	Mat imgEdged; // = imgSource.clone();
	imgEdged = imgGrad & imgCanny;
	imshow("Edged", imgEdged);

	Mat imgDistanceTr;
	distanceTransform(imgEdged, imgDistanceTr, CV_DIST_L2, 3);
	normalize(imgDistanceTr, imgDistanceTr, 0, 1., NORM_MINMAX);
	imshow("distanceTransform", imgDistanceTr);
	
	Mat imgThresholdDTr;
	// Threshold to obtain the peaks
	// This will be the markers for the foreground objects
	threshold(imgDistanceTr, imgDistanceTr, (double)pDistThreshold / 100, 1., CV_THRESH_BINARY);
	// Dilate a bit the dist image
	Mat kernel1 = Mat::ones(3, 3, CV_8UC1);
	dilate(imgDistanceTr, imgThresholdDTr, kernel1);
	imshow("imgThresholdDTr", imgThresholdDTr);


	//morphological opening (remove small objects from the foreground)
	Mat imgMorph;
	if (iOE != 0)
		erode(imgDistanceTr, imgMorph, getStructuringElement(MORPH_ELLIPSE, Size(iOE, iOE)));
	if (iOD != 0)
		dilate(imgMorph, imgMorph, getStructuringElement(MORPH_ELLIPSE, Size(iOD, iOD)));

	//morphological closing (fill small holes in the foreground)
	if (iCD != 0)
		dilate(imgMorph, imgMorph, getStructuringElement(MORPH_ELLIPSE, Size(iCD, iCD)));
	if (iCE != 0)
		erode(imgMorph, imgMorph, getStructuringElement(MORPH_ELLIPSE, Size(iCE, iCE)));

	imshow("morphological", imgMorph);


	// Create the CV_8U version of the distance image
	// It is needed for findContours()
	Mat dist_8u;
	imgMorph.convertTo(dist_8u, CV_8U);
	// Find total markers
	vector<vector<Point> > contours;
	findContours(dist_8u, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	// Create the marker image for the watershed algorithm
	Mat markers = Mat::zeros(imgDistanceTr.size(), CV_32SC1);

	Mat imgBoundingRect = Mat::zeros(imgDistanceTr.size(), imgSource.type());
	// Draw the foreground markers
	for (int i = 0; i < contours.size(); i++) {

		//vector<Point> poly;
		
		// eliminate the blobs that contact with edge
		//approxPolyDP(Mat(contours[i]), poly, 3, true);

		Rect bRect = boundingRect(contours[i]);
		bool isTouchEdge = (
			bRect.tl().x <= 1 
			|| bRect.tl().y <= 1 
			|| bRect.br().x >= imgSource.size().width - 1 
			|| bRect.br().y >= imgSource.size().height - 1
			);
		Scalar _color = isTouchEdge ? Scalar(255, 0, 0) : Scalar(255, 255, 255);

		if (!isTouchEdge) {
			drawContours(markers, contours, static_cast<int>(i), Scalar::all(static_cast<int>(i)+1), -1);
			//printf("not touched edge (%d,%d), (%d,%d)\n", bRect.tl().x, bRect.tl().y, bRect.br().x, bRect.br().y);
		}
		else {
			//printf("touched edge (%d,%d), (%d,%d)\n", bRect.tl().x, bRect.tl().y, bRect.br().x, bRect.br().y);
		}
		
		rectangle(imgBoundingRect, bRect, _color);

	}
	// Draw the background marker
	circle(markers, Point(5, 5), 3, CV_RGB(255, 255, 255), -1);
	imshow("Markers", markers * 10000);
	imshow("imgBoundingRect", imgBoundingRect);

	//printf("contours count=%d\n", contours.size());


	// Perform the watershed algorithm
	watershed(imgSource, markers);
	
	Mat mark = Mat::zeros(markers.size(), CV_8UC1);
	markers.convertTo(mark, CV_8UC1);
	
	bitwise_not(mark, mark);
	//    imshow("Markers_v2", mark); // uncomment this if you want to see how the mark

	int nContoursOf4 = 0;

	vector< vector<Point> > contours_m;
	findContours(Mat(markers), contours_m, CV_RETR_FLOODFILL, CV_CHAIN_APPROX_SIMPLE);
	
	vector< c_a_pair > goodContoursPaired;

	Mat imgPoly4Filled = Mat::zeros(imgSource.size(), CV_32F);

	vector<Point> points_all_approx, points_all_notApprox;
	vector<Point> arr_moment;
	vector<dcpPair> arr_dcpPair;

	Mat imgContoursWaterShed = Mat::zeros(imgSource.size(), imgSource.type());
	for (int i = 0; i < contours_m.size(); i++) {
		vector<Point> poly;

		double epsilon = (double)0.02 * arcLength(contours_m[i], true);

		approxPolyDP(Mat(contours_m[i]), poly,
			4, // accuracy of the approximation  
			true); // yes it is a closed shape  

		if (poly.size() == 4 && isContourConvex(poly)) {

			double area = contourArea(poly);
			if (area > 0) {
				c_a_pair paired(poly, area);
				goodContoursPaired.push_back(paired);

				polylines(imgContoursWaterShed, poly, true, Scalar(0, 255, 0), 2);
				nContoursOf4++;

				fillConvexPoly(imgPoly4Filled, poly, Scalar::all(1));

				for each (Point p in poly)
				{
					points_all_approx.push_back(p);
				}
				for each (Point p in contours_m[i])
				{
					points_all_notApprox.push_back(p);
				}

				// calc the centre point
				Moments m = moments(contours_m[i], false);
				Point pCentre(m.m10 / m.m00, m.m01 / m.m00);
				arr_moment.push_back(pCentre);

				// make distance-centrePoint-polyArr pair
				dcpPair newDcpPair;
				newDcpPair.centrePoint = pCentre;
#define sqr(x) x*x
				newDcpPair.distance = sqrt(sqr(pCentre.x) + sqr(pCentre.y));
				newDcpPair.poly = tPoly(std::begin(contours_m[i]), std::end(contours_m[i]));

				arr_dcpPair.push_back(newDcpPair);
			}
		}
		else {
			polylines(imgContoursWaterShed, poly, true, Scalar(155, 155, 155), 0.5);
		}
	}
	imshow("Contours WaterShed", imgContoursWaterShed);
	printf("nContoursOf4 = %d\n", nContoursOf4);

	imshow("Poly4Filled", imgPoly4Filled);

	Mat imgPolyGrouping = imgPoly4Filled.clone();

	/// convert to grayScale (from binary)
	//imgPolyGrouping *= 255;
	//imgPolyGrouping.convertTo(imgPolyGrouping, CV_GRAY2BGR);

	/*
	RotatedRect rRect = minAreaRect(points_all_approx);
	Point2f pointsRect[4];
	rRect.points(pointsRect);
	line(imgPolyGrouping, pointsRect[0], pointsRect[1], Scalar(155, 155, 155), 1);
	line(imgPolyGrouping, pointsRect[1], pointsRect[2], Scalar(155, 155, 155), 2);
	line(imgPolyGrouping, pointsRect[2], pointsRect[3], Scalar(155, 155, 155), 1);
	line(imgPolyGrouping, pointsRect[3], pointsRect[0], Scalar(155, 155, 155), 2);
	*/

	/*
	vector<Point> points_convexHull;
	convexHull(points_all_approx, points_convexHull);
	//drawContours(imgPolyGrouping, points_convexHull, -1, Scalar(155, 155, 155));
	fillConvexPoly(imgPolyGrouping, points_convexHull, Scalar::all(1));//Scalar(155, 155, 155));

	imgPolyGrouping = imgPolyGrouping - imgPoly4Filled;
	*/

	/// TODO:
	// 1. make centrePoint-Poly pair

	// 2. sort those pairs according to the distance between (0,0) and centrePoint
	arr_dcpPair = sortPairedGroupByDistance(arr_dcpPair);

	// for a test, we draw all the centrePoint
	Mat imgSortedCentrePoint = Mat::zeros(imgSource.size(), imgSource.type());
	int i = 0;
	for each (dcpPair oneDcpPair in arr_dcpPair)
	{
		circle(imgSortedCentrePoint, oneDcpPair.centrePoint, 2, Scalar(255, 0, 0));
		putText(imgSortedCentrePoint, std::to_string(i++), oneDcpPair.centrePoint,
			FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, Scalar(255, 0, 0));
	}
	//imshow("imgSortedCentrePoint", imgSortedCentrePoint);

	// 3. go thru the sorted pairs,  ... described on my draft paper
	Mat img4x4DetectPoint(imgSource.size(), imgSource.type());
	Mat imgPoly, imgPolyResult;

	const int maxFindCount = 6;
	bool endOfFind_AD_Pair = false;
	for (std::vector<dcpPair>::reverse_iterator rit = arr_dcpPair.rbegin();
		rit != arr_dcpPair.rend() && (rit - arr_dcpPair.rbegin()) < maxFindCount;
		++rit) {

		// reverse find the D

		for (std::vector<dcpPair>::iterator it = arr_dcpPair.begin();
			it != arr_dcpPair.end() && (it - arr_dcpPair.begin()) < maxFindCount;
			++it) {

			// find the A

			double angle = cvFastArctan(
				(rit->centrePoint.y - it->centrePoint.y),
				(rit->centrePoint.x - it->centrePoint.x));
			//line(imgSortedCentrePoint, it->centrePoint, rit->centrePoint, Scalar(0, 255, 0));

			float lineLength = euclideanDist(it->centrePoint, rit->centrePoint);
			double gap = lineLength / sqrt(2) / 3;

			get4x4PointsDetectionMat(img4x4DetectPoint, it->centrePoint, 45-angle, gap);

			// chech if we have a good match with detect points
			bool endOfDetect = false;
			int goodPieceCount = 0;
			for (std::vector<dcpPair>::iterator itPoly = arr_dcpPair.begin();
				itPoly != arr_dcpPair.end();
				++itPoly) {

				// draw one poly

				imgPoly = Mat::zeros(imgSource.size(), CV_8U);				
				fillConvexPoly(imgPoly, itPoly->poly, Scalar::all(255));

				bitwise_and(imgPoly, img4x4DetectPoint, imgPolyResult);
				int nPoints = countNonZero(imgPolyResult);
				//printf("nPoints=%d\n", nPoints);

				switch (nPoints)
				{
				case 0:
					// maybe blob of poly is not the piece in 4x4, continue to find
					continue;
				case 1:
					// good! this seem to be a one piece of the 4x4

					// then we eliminate this point
					bitwise_xor(imgPolyResult, img4x4DetectPoint, img4x4DetectPoint);

					/// TODO: record this poly, because he is damn good

					goodPieceCount++;

					continue;

				default:
					// when count > 1,
					//		this A-D pair is not good, continue to find A-D pair...
					endOfDetect = true;
					break;
				}


				if (endOfDetect)					
					break;

			
				//break; // for debug

			}

			printf("we have goodPieceCount in this A-D pair: %d\n", goodPieceCount);

			if (endOfFind_AD_Pair)
				break;
			
			//break; // for debug

		}
		if (endOfFind_AD_Pair)
			break;

		//break; // for debug
	}

	//imshow("imgPoly", imgPoly);
	//imshow("img4x4DetectPoint", img4x4DetectPoint);
	imshow("imgSortedCentrePoint", imgSortedCentrePoint);


	return;



	/// convert to grayScale (from binary)
	imgPolyGrouping *= 255;
	imgPolyGrouping.convertTo(imgPolyGrouping, CV_GRAY2BGR);

	//distanceTransform(imgPolyGrouping, imgPolyGrouping, CV_DIST_L2, 3);
	//normalize(imgPolyGrouping, imgPolyGrouping, 0, 1, NORM_MINMAX);

	imshow("imgPolyGrouping", imgPolyGrouping);




	/// calc group
	printAreas(goodContoursPaired);
	vector<c_a_pair> sortedGroup = sortPairedGroupByArea(goodContoursPaired);
	//printAreas(sortedGroup);

	vector<vector<c_a_pair>> groupingGroups = sortedGrouping_continous(sortedGroup, pSortedGroupingGap);
	//printGroups(groupingGroups);





	// image looks like at that point
	// Generate random colors
	vector<Vec3b> colors;
	for (size_t i = 0; i < contours.size(); i++)
	{
		int b = theRNG().uniform(0, 255);
		int g = theRNG().uniform(0, 255);
		int r = theRNG().uniform(0, 255);
		colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
	}
	// Create the result image
	Mat dst = Mat::zeros(markers.size(), CV_8UC3);
	// Fill labeled objects with random colors
	for (int i = 0; i < markers.rows; i++)
	{
		for (int j = 0; j < markers.cols; j++)
		{
			int index = markers.at<int>(i, j);
			if (index > 0 && index <= static_cast<int>(contours.size()))
				dst.at<Vec3b>(i, j) = colors[index - 1];
			else
				dst.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
		}
	}
	// Visualize the final image
	imshow("Colored Watershed", dst);


	

	return;



}



int key_1;
int main() {

	init();

	// test sortedGrouping() 
	//vector < int > vList { 5, 8, 8, 21, 25, 26, 28, 40, 43, 55 };
	//printGroups( sortedGrouping(vList, 2) );



#if 0

	//20160526_155743
	//20160526_155725
	// bad 20160526_155723
	imgSource = imread("D:\\WorkSpace\\WIN\\VS2013\\CSharp\\opencv_cube_recognize\\raw\\20160526_155743.jpg");

	//resize(imgSource, imgSource, Size(0, 0), 0.2, 0.2);
	resizeToLong(imgSource, 350);
	imshow("src img(resized)", imgSource);	

	showImg();
#else
	// init capture
	VideoCapture cap;
	cap.open(1);

	// create new window  to show image

	while (1) {
		cap >> imgSource;

		resizeToLong(imgSource, 350);

		// print the image;
		imshow("raw", imgSource);
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
