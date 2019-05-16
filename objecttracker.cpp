#include "objecttracker.h"
#include <iostream>

using namespace std;
using namespace cv;


ObjectTracker::ObjectTracker() {
	nextID = 0;
	maxDisappeared = 50;
	maxDistance = 50;
}


ObjectTracker::register(Point p, bool state) {
	object[nextID] = p;
	states[nextID] = state;
	disappeared[nextID] = 0;
	nextID++;
}


ObjectTracker::deregister(int ID) {
	object.erase(ID);
	states.erase(ID)l
	disappeared.erase(ID);
}