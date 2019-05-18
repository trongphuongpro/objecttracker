#ifndef GUARD_OBJECTTRACKER_H
#define GUARD_OBJECTTRACKER_H

#include <vector>
#include <map>
#include <set>
#include <string>
#include "opencv2/imgproc.hpp"
#include "opencv2/tracking/tracker.hpp"

class ObjectTracker {
public:
	ObjectTracker(int = 30);
	std::vector<cv::Rect> update(const cv::Mat&, 
				bool = true,
				std::vector<cv::Rect> = std::vector<cv::Rect>());

	std::map<int, cv::Point> getCentroids();

	std::map<int, bool> getStates();
	void setState(int, bool);

	void showInfo();
	
private:
	int nextID;
	int maxDisappeared;
	int maxDistance;
	
	std::map<int, cv::Point> objects;
	std::map<int, bool> states;
	std::map<int, int> disappeared;
	std::map<int, cv::Ptr<cv::Tracker>> trackers;

	void registerObject(const cv::Mat&, const cv::Rect2d&, bool);
	void deregisterObject(int);
};

cv::Mat cdist(const std::vector<cv::Point>&, const std::vector<cv::Point>&);

template<typename T>
std::vector<size_t> argsort(const std::vector<T>&);

template<typename T>
size_t argmin(const std::vector<T>&);

void printSet(const std::set<size_t>&, const std::string&);
void printVector(const std::vector<size_t>&, const std::string&);
void printBox(const std::vector<cv::Point>&, const std::string&);

void test(const std::vector<cv::Point>&, const std::vector<cv::Point>&);
#endif