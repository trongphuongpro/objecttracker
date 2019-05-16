#ifndef GUARD_OBJECTTRACKER_H
#define GUARD_OBJECTTRACKER_H

#include <vector>
#include <map>
#include "opencv2/imgproc.hpp"
#include "opencv2/tracking/tracker.hpp"

class ObjectTracker {
public:
	ObjectTracker();
	std::vector<cv::Rect> update(std::vector<cv::Rect>);

private:
	int nextID;
	int maxDisappeared;
	int maxDistance;
	
	std::map<int, cv::Point> objects;
	std::map<int, bool> state;
	std::map<int, int> disappeared;
	std::vect<cv::Ptr<TrackerKCF>> trackers;

	void register(cv::Point, bool);
	void deregister(int);
};

#endif