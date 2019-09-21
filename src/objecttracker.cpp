#include "objecttracker.h"
#include <iostream>
#include <algorithm>

using namespace std;
using namespace cv;


ObjectTracker::ObjectTracker(int _maxFrame) {
	nextID = 0;
	maxDisappeared = _maxFrame;
	maxDistance = 100;
}


void ObjectTracker::addTracker(const Mat& frame, const Rect& roi, int ID) {
	Ptr<TrackerKCF> tracker = TrackerKCF::create();
	bool ret = tracker->init(frame, roi);

	if (ret) {
		#ifdef DEBUG	
		cout << ">> Init successfully" << endl;
		#endif
	}
	
	trackers[ID] = tracker;
}


void ObjectTracker::removeTracker(int ID) {
	trackers[ID].release();

	#ifdef DEBUG
		if (trackers[ID].empty()) {
			cout << ">> Tracker has been deallocated." << endl;
		}
	#endif
}


void ObjectTracker::registerObject(const Mat& frame, const Rect& roi, bool state) {
	#ifdef DEBUG
		cout << "Register new object...";
		cout << "ID: " << nextID << endl;
	#endif

	Point p(roi.x + roi.width/2, roi.y + roi.height/2);
	objects[nextID] = p;
	states[nextID] = state;

	addTracker(frame, roi, nextID);

	disappeared[nextID] = 0;
	nextID++;

	#ifdef DEBUG
		cout << "[Done]" << endl;
	#endif
}


void ObjectTracker::deregisterObject(int ID) {
	#ifdef DEBUG
		cout << "Deregister object ID: " << ID << endl;
	#endif

	removeTracker(ID);
	trackers.erase(ID);
	objects.erase(ID);
	states.erase(ID);
	disappeared.erase(ID);
	
	#ifdef DEBUG
		cout << "[Done]" << endl;
	#endif
}


void ObjectTracker::update(const Mat& frame, vector<Rect>& boxes) {

	#ifdef DEBUG 
		cout << "Boxes: " << boxes.size() << endl;
	#endif

	updateCentroids(frame, boxes, false);
}


vector<Rect> ObjectTracker::update(const Mat& frame) {
	#ifdef DEBUG
			cout << "[Use tracker estimation]" << endl;
	#endif

	vector<Rect> boxes;

	for (const auto& e : trackers) {
		Rect2d roi;
		bool ret = e.second->update(frame, roi);

		if (ret) {
				boxes.push_back(roi);
		}
	}

	#ifdef DEBUG 
		cout << "Boxes: " << boxes.size() << endl;
	#endif

	updateCentroids(frame, boxes, true);

	return boxes;
}	



void ObjectTracker::updateCentroids(const Mat& frame, vector<Rect>& boxes, bool useTracker) {

	if (boxes.empty()) {
		#ifdef DEBUG
			cout << "Nothing detected" << endl;
		#endif

		for (auto& e : disappeared) {
			e.second++;
			if (e.second > maxDisappeared) {
				deregisterObject(e.first);
			}
		}
	}

	else if (objects.empty()) {
		for (const auto& e : boxes) {
			registerObject(frame, e, false);
		}
	}
	else {
		vector<Point> inputCentroids;

		for (const auto& e : boxes) {
			inputCentroids.push_back(Point(e.x + e.width/2, e.y + e.height/2));
		}

		vector<int> objectIDs;
		vector<Point> objectCentroids;

		for (const auto& e : objects) {
			objectIDs.push_back(e.first);
			objectCentroids.push_back(e.second);
		}

		#ifdef DEBUG
			printBox(objectCentroids, "objectCentroids");
			printBox(inputCentroids, "inputCentroids");
		#endif

		Mat D = cdist(objectCentroids, inputCentroids);

		vector<size_t> rows;
		vector<size_t> cols;
		vector<size_t> temp;
		for (int i = 0; i < D.size().height; i++) {
				
			vector<size_t> row;
			vector<size_t> col;
			
			for (int j = 0; j < D.size().width; j++) {
				row.push_back(D.at<int32_t>(i, j));
				col.push_back(D.at<int32_t>(i, j));
			}

			sort(row.begin(), row.end());
			rows.push_back(row[0]);
			temp.push_back(argmin(col));
		}

		rows = argsort(rows);

		for (size_t i = 0; i < rows.size(); i++) {
			cols.push_back(temp[rows[i]]);
		}

		#ifdef DEBUG
			printVector(rows, "rows");
			printVector(cols, "cols");
		#endif

		vector<vector<int>> coord;
		for (size_t i = 0; i < rows.size(); i++) {
			vector<int> c;
			c.push_back(rows[i]);
			c.push_back(cols[i]);
			coord.push_back(c);
		}

		set<size_t> usedRows;
		set<size_t> usedCols;

		for (const auto& e : coord) {
			size_t row = e[0];
			size_t col = e[1];

			#ifdef DEBUG
				cout << endl << "row: " << row << " col: " << col << " ";
			#endif

			if ((usedRows.find(row) != usedRows.end()) 
				|| (usedCols.find(col) != usedCols.end())) {

				#ifdef DEBUG
					cout << ">> used" << endl;
				#endif

				continue;
			}

			int ID = objectIDs[row];
			if (D.at<int32_t>(row, col) < maxDistance*maxDistance) {
				objects[ID] = inputCentroids[col];
				disappeared[ID] = 0;

				if (!useTracker) {
					removeTracker(ID);

					addTracker(frame, boxes[col], ID);
				}

				usedRows.insert(row);
				usedCols.insert(col);
			}
		}

		#ifdef DEBUG
			printSet(usedRows, "usedRows");
			printSet(usedCols, "usedCols");
		#endif

		set<size_t> unusedRows;
		for (int i = 0; i < D.size().height; i++) {
			if (usedRows.find(i) == usedRows.end()) {
				unusedRows.insert(i);
			}
		}

		set<size_t> unusedCols;
		for (int i = 0; i < D.size().width; i++) {
			if (usedCols.find(i) == usedCols.end()) {
				unusedCols.insert(i);
			}
		}

		#ifdef DEBUG
			printSet(unusedRows, "unusedRows");
			printSet(unusedCols, "unusedCols");
		#endif
		
		// some objects disappeared
		for (const auto& row : unusedRows) {
			int ID = objectIDs[row];
			disappeared[ID]++;

			if (disappeared[ID] > maxDisappeared) {
				deregisterObject(ID);
			}
		}
	
		// some new objects will be registered
		if (!useTracker) {
			for (const auto& col : unusedCols) {
				registerObject(frame, boxes[col], false);
			}
		}
	}
}



map<int, Point> ObjectTracker::getCentroids() {
	return objects;
}


map<int, bool> ObjectTracker::getStates() {
	return states;
}

void ObjectTracker::setState(int ID, bool value) {
	states.at(ID) = value;
}


template<typename T>
vector<size_t> argsort(const vector<T>& input) {
	vector<size_t> idxs(input.size());
	iota(idxs.begin(), idxs.end(), 0);
	    
	sort(idxs.begin(), 
	  	idxs.end(), 
	  	[&](size_t a, size_t b) {
	        return input[a] < input[b];
	    });
	                
	return idxs;
}


template<typename T>
size_t argmin(const vector<T>& input) {

	return argsort(input)[0];
}


Mat cdist(const vector<Point>& A, const vector<Point>& B) {
	Mat result(A.size(), B.size(), CV_32SC1);

	for (size_t r = 0; r < A.size(); r++) {
		for (size_t c = 0; c < B.size(); c++) {
			result.at<uint32_t>(r, c) = (A[r].x - B[c].x)*(A[r].x - B[c].x)
									  + (A[r].y - B[c].y)*(A[r].y - B[c].y);
		}
	}
	return result;
}


void ObjectTracker::showInfo() {
	cout << "#Trackers: " << trackers.size() << endl;
	cout << "#Objects: " << objects.size() << endl;
	cout << "------------------------------------------" << endl;
}


void printSet(const set<size_t>& v, const string& name) {
	cout << name << ": ";
	for (const auto& e : v) {
		cout << e << " ";
	}
	cout << endl;
}


void printVector(const vector<size_t>& v, const string& name) {
	cout << name << ": ";
	for (const auto& e : v) {
		cout << e << " ";
	}
	cout << endl;
}


void printBox(const vector<Point>& v, const string& name) {
	cout << name << ": ";
	for (const auto& e : v) {
		cout << e.x << " " << e.y << endl;
	}
	cout << endl;
}