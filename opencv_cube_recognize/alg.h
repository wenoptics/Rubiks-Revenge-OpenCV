#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>


using namespace cv;
using namespace std;

#define sqr(x) x*x

typedef pair<vector<Point>, double> c_a_pair;

vector<vector<c_a_pair>> sortedGrouping(vector<c_a_pair> list, double maxGap);
vector<vector<c_a_pair>> sortedGrouping_continous(vector<c_a_pair> list, double maxGap);
void printGroups(vector<vector<c_a_pair>> groups);


vector<c_a_pair> makeContourAreaPair(vector<vector<Point> > contours);
vector<c_a_pair> sortPairedGroupByArea(vector<c_a_pair> pairedGroup);
void printAreas(vector<c_a_pair> pairedGroup);

void resizeToLong(Mat& img, int longLength);


/// 

typedef vector<Point> tPoly;

//distance-centrePoint-polyArr pair
typedef struct {
	double distance;
	Point centrePoint;
	tPoly poly;
} dcpPair;

bool cmp_sortPairedGroupByDistance(const dcpPair& a, dcpPair& b);
vector<dcpPair> sortPairedGroupByDistance(vector<dcpPair> pairedGroup);

float euclideanDist(Point& p, Point& q);