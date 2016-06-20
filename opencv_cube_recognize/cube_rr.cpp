#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include "alg.h"

using namespace cv;
using namespace std;

#define useStaticImage
#undef useStaticImage

void process_frame();
static void on_trackbar(int, void*) {
	process_frame();
}

int paraCannyT1 = 167;
int paraCannyT2 = 340;
int paraOE = 12; 
int paraOD = 11; 
int paraCE = 9; 
int paraCD = 1;
int paraDistThreshold = 19;

void init() {

	namedWindow("Control", CV_WINDOW_NORMAL);

	createTrackbar("pCannyT1", "Control", &paraCannyT1, 1000, on_trackbar);
	createTrackbar("pCannyT2", "Control", &paraCannyT2, 1000, on_trackbar);
	
	createTrackbar("mOpen erode", "Control", &paraOE, 30, on_trackbar);
	createTrackbar("mOpen dilate", "Control", &paraOD, 30, on_trackbar);
	createTrackbar("mClose erode", "Control", &paraCE, 30, on_trackbar);
	createTrackbar("mClose dilate", "Control", &paraCD, 30, on_trackbar);
	
	createTrackbar("DistThreshold", "Control", &paraDistThreshold, 100, on_trackbar);

}

Mat imgSource;
void process_frame() {
	
	/// convert imgage to grayscale
	Mat imgGray;
	cvtColor(imgSource, imgGray, CV_BGR2GRAY);


	/// do canny, find edge
	Mat imgCanny;
	Canny(imgGray, imgCanny, paraCannyT1, paraCannyT2, 3);
	imshow("Canny", imgCanny);


	/// inverse the canny output, (make edges to black lines)
	bitwise_not(imgCanny, imgCanny);
	// this can also do it: threshold(imgCanny, imgCanny, 1, 255, CV_THRESH_BINARY_INV);
	imshow("Inversed Canny", imgCanny);


	/// cover the black edges on the origin image
	Mat imgEdged; // = imgSource.clone();
	imgEdged = imgGray & imgCanny;
	imshow("overlaped with edge", imgEdged);


	/// do distanceTransform
	Mat imgDistanceTr;
	distanceTransform(imgEdged, imgDistanceTr, CV_DIST_L2, 3);
	normalize(imgDistanceTr, imgDistanceTr, 0, 1., NORM_MINMAX);
	imshow("distanceTransform", imgDistanceTr);
	

	/// obtain the peak
	// Threshold to obtain the peaks
	Mat imgThresholdDTr;
	// This will be the markers for the foreground objects
	threshold(imgDistanceTr, imgDistanceTr, (double)paraDistThreshold / 100, 1., CV_THRESH_BINARY);
	// Dilate a bit the dist image
	Mat kernel1 = Mat::ones(3, 3, CV_8UC1);
	dilate(imgDistanceTr, imgThresholdDTr, kernel1);
	imshow("peak of distanceTransform", imgThresholdDTr);


	/// morphological opening (remove small objects from the foreground)
	Mat imgMorph;
	if (paraOE != 0)
		erode(imgDistanceTr, imgMorph, getStructuringElement(MORPH_ELLIPSE, Size(paraOE, paraOE)));
	if (paraOD != 0)
		dilate(imgMorph, imgMorph, getStructuringElement(MORPH_ELLIPSE, Size(paraOD, paraOD)));

	//morphological closing (fill small holes in the foreground)
	if (paraCD != 0)
		dilate(imgMorph, imgMorph, getStructuringElement(MORPH_ELLIPSE, Size(paraCD, paraCD)));
	if (paraCE != 0)
		erode(imgMorph, imgMorph, getStructuringElement(MORPH_ELLIPSE, Size(paraCE, paraCE)));

	imshow("morphological ", imgMorph);


	// Create the CV_8U version of the distance image
	// It is needed for findContours()
	Mat dist_8u;
	imgMorph.convertTo(dist_8u, CV_8U);


	// Find total markers
	vector<vector<Point> > contours;
	findContours(dist_8u, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	// Create the marker image for the watershed algorithm
	Mat markers = Mat::zeros(imgDistanceTr.size(), CV_32SC1);

	// this Matrix is for debug print
	Mat imgBoundingRect = Mat::zeros(imgDistanceTr.size(), imgSource.type()); 

	// Draw the foreground markers
	for (int i = 0; i < contours.size(); i++) {

		//vector<Point> poly;

		/// eliminate the blobs that contact with edge
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

		rectangle(imgBoundingRect, bRect, _color); // this is for debug

	}
	// Draw the background marker
	circle(markers, Point(5, 5), 3, CV_RGB(255, 255, 255), -1);
	
	imshow("Markers", markers * 10000);
	imshow("imgBoundingRect", imgBoundingRect); // this is for debug

	
	// Perform the watershed algorithm
	watershed(imgSource, markers);

	Mat mark = Mat::zeros(markers.size(), CV_8UC1);
	markers.convertTo(mark, CV_8UC1);

	bitwise_not(mark, mark);
	//    imshow("Markers_v2", mark); // uncomment this if you want to see how the mark
	
	/// find all the 4-contour polygons
	int nContoursOf4 = 0;

	vector< vector<Point> > contours_m;
	findContours(Mat(markers), contours_m, CV_RETR_FLOODFILL, CV_CHAIN_APPROX_SIMPLE);
	
	//Mat imgPoly4Filled = Mat::zeros(imgSource.size(), CV_32F);

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

				polylines(imgContoursWaterShed, poly, true, Scalar(0, 255, 0), 2);
				nContoursOf4++;

				//fillConvexPoly(imgPoly4Filled, poly, Scalar::all(1));

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
				newDcpPair.distance = sqrt(sqr(pCentre.x) + sqr(pCentre.y));
				newDcpPair.poly = tPoly(std::begin(contours_m[i]), std::end(contours_m[i]));

				arr_dcpPair.push_back(newDcpPair);
			}
		}
		else {
			polylines(imgContoursWaterShed, poly, true, Scalar(155, 155, 155), 0.5);
		}
	}
	imshow("Contours(4) WaterShed", imgContoursWaterShed);
	printf("nContoursOf4 = %d\n", nContoursOf4);

	//imshow("Poly4Filled", imgPoly4Filled);
	
	
	/// sort those pairs according to the distance between (0,0) and centrePoint
	arr_dcpPair = sortPairedGroupByDistance(arr_dcpPair);


	// for testing purpose, we draw all the centrePoint
	Mat imgSortedCentrePoint = Mat::zeros(imgSource.size(), imgSource.type());
	int i = 0;
	for each (dcpPair oneDcpPair in arr_dcpPair)
	{
		circle(imgSortedCentrePoint, oneDcpPair.centrePoint, 2, Scalar(255, 0, 0));
		putText(imgSortedCentrePoint, std::to_string(i++), oneDcpPair.centrePoint,
			FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, Scalar(255, 0, 0));
	}
	imshow("imgSortedCentrePoint", imgSortedCentrePoint);
		

	Mat img4x4DetectPoint; // (imgSource.size(), imgSource.type());
	Mat imgPoly, imgPolyResult;

	const int maxFindCount = 6;
	bool endOfFind_AD_Pair = false;

	for (std::vector<dcpPair>::reverse_iterator rit = arr_dcpPair.rbegin();
		rit != arr_dcpPair.rend();
		++rit) {

		// reverse find the D (rit)

		int countOfD = rit - arr_dcpPair.rbegin();
		if (countOfD >= maxFindCount)
			/// This is another end-of-loop condition, just in case of find too much (consuming too much time)
			break;

		for (std::vector<dcpPair>::iterator it = arr_dcpPair.begin();
			it != arr_dcpPair.end();
			++it) {

			// find the A (it)

			int countOfA = it - arr_dcpPair.begin();
			if (countOfA >= maxFindCount)
				/// This is another end-of-loop condition, just in case of find too much (consuming too much time)
				break;


			/// calc the characteristic of this A-D pair (i.e. the angle and the gap)
			double angle = fastAtan2(
				(rit->centrePoint.y - it->centrePoint.y),
				(rit->centrePoint.x - it->centrePoint.x));
			//line(imgSortedCentrePoint, it->centrePoint, rit->centrePoint, Scalar(0, 255, 0));

			float lineLength = euclideanDist(it->centrePoint, rit->centrePoint);
			double gap = lineLength / sqrt(2) / 3;

			printf("[i]got A-D pair [%d]-[%d] (of %d) with angle=%.2f, gap=%.2f \n",
				countOfA, countOfD, arr_dcpPair.size(), angle, gap);

			/*
			get4x4PointsDetectionMat(img4x4DetectPoint, it->centrePoint, 45 - angle, gap);

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
			*/

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


}


int main() {

	init();
	
#ifdef useStaticImage

	//20160526_155743
	//20160526_155725
	// bad 20160526_155723
	imgSource = imread("D:\\WorkSpace\\WIN\\VS2013\\CSharp\\opencv_cube_recognize\\raw\\20160526_155743.jpg");

	//resize(imgSource, imgSource, Size(0, 0), 0.2, 0.2);
	resizeToLong(imgSource, 350);
	imshow("src img(resized)", imgSource);

	process_frame();
	
	while (true){
		if (waitKey() == 27){
			break;
		}
	}

#else
	// init capture
	VideoCapture cap;
	cap.open(1);

	// create new window  to show image

	while (1) {
		cap >> imgSource;

		resizeToLong(imgSource, 350);

		// show the image;
		imshow("raw(resized)", imgSource);

		process_frame();

		// delay 33ms
		waitKey(22);

	}

#endif

	return 0;

}
