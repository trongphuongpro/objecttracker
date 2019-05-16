#include "objecttracker.h"
#include <iostream>

using namespace std;
using namespace cv;


ObjectTracker::ObjectTracker() {
	nextID = 0;
	maxDisappeared = 50;
	maxDistance = 50;
}


