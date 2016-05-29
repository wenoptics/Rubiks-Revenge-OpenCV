#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include "alg.h"



using namespace cv;
using namespace std;


bool cmp(const c_a_pair& a, c_a_pair& b) {
	return a.second < b.second;
}

void printAreas(vector<c_a_pair> pairedGroup) {
	for each (c_a_pair paired in pairedGroup)
	{
		printf("%.2f, ", paired.second);
	}
	printf("\n\n");
}

vector<c_a_pair> sortPairedGroupByArea(vector<c_a_pair> pairedGroup) {
	sort(pairedGroup.begin(), pairedGroup.end(), cmp);
	return pairedGroup;
}

vector<c_a_pair> makeContourAreaPair(vector<vector<Point> > contours) {
	
	vector<c_a_pair> pairedGroup;
	
	for (int i = 0; i < contours.size(); i++) {
		c_a_pair paired(contours[i], contourArea(contours[i]));
		pairedGroup.push_back(paired);
	}

	return pairedGroup;

}



vector<vector<c_a_pair>> sortedGrouping_continous(vector<c_a_pair> list, double maxGap) {

	vector<vector<c_a_pair>> groups;

	if (list.size() < 2) return groups;
	
	vector<c_a_pair> newGroup;
	
	int nowIndex;
	for (nowIndex = 0;; nowIndex++)
	{
		newGroup.push_back(list[nowIndex]);

		if (nowIndex + 1 == list.size()) { 
			groups.push_back(newGroup); 
			break;
		} else{
			if (list[nowIndex + 1].second - list[nowIndex].second < maxGap) {
				continue;
			}
			else {
				groups.push_back(newGroup);
				newGroup = vector<c_a_pair>();
			}
		}		
	}

	return groups;

}

vector<vector<c_a_pair>> sortedGrouping(vector<c_a_pair> list, double maxGap) {

	vector<vector<c_a_pair>> groups;

	if (list.size() == 0) return groups;

	int indexStart = 0;

	while (1) {
		vector<c_a_pair> newGroup;
		double startingVal = list[indexStart].second;
		bool hasNext = false;

		for (int nowIndex = indexStart; nowIndex < list.size(); nowIndex++)
		{
			double ele = list[nowIndex].second;

			if (ele - startingVal < maxGap) {
				newGroup.push_back(list[nowIndex]);
			}
			else {
				indexStart = nowIndex;
				hasNext = true;
				break;
			}
		}

		groups.push_back(newGroup);

		if (!hasNext)
		{
			break;
		}
	}

	return groups;

}

void printGroups(vector<vector<c_a_pair>> groups) {
	int nGroup = 0;
	for each (vector<c_a_pair> oneGroup in groups)
	{
		printf("group %d:\n", nGroup++);
		for each (c_a_pair ele in oneGroup)
		{
			printf("%.2f, ", ele.second);
		}
		printf("\n\n");
	}
}

