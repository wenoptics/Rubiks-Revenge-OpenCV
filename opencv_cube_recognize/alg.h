#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>


using namespace cv;
using namespace std;


typedef pair<vector<Point>, double> c_a_pair;

vector<vector<c_a_pair>> sortedGrouping(vector<c_a_pair> list, double maxGap);
vector<vector<c_a_pair>> sortedGrouping_continous(vector<c_a_pair> list, double maxGap);
void printGroups(vector<vector<c_a_pair>> groups);


vector<c_a_pair> makeContourAreaPair(vector<vector<Point> > contours);
vector<c_a_pair> sortPairedGroupByArea(vector<c_a_pair> pairedGroup);
void printAreas(vector<c_a_pair> pairedGroup);