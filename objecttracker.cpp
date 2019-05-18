#include "objecttracker.h"
#include <iostream>
#include <algorithm>
#include <set>

using namespace std;
using namespace cv;


ObjectTracker::ObjectTracker() {
	nextID = 0;
	maxDisappeared = 30;
	maxDistance = 50;
}


void ObjectTracker::registerObj(const Mat& frame, const Rect2d& roi, bool state) {
	cout << "Register new object...";

	Point p(roi.x + roi.width/2, roi.y + roi.height/2);
	objects[nextID] = p;

	states[nextID] = state;

	Ptr<TrackerKCF> tracker = TrackerKCF::create();
	tracker->init(frame, roi);
	trackers[nextID] = tracker;

	disappeared[nextID] = 0;
	nextID++;
	cout << "[Done]" << endl;
}


void ObjectTracker::deregister(int ID) {
	cout << "Deregister object " << ID << endl;

	trackers[ID].release();
	if (trackers[ID].empty()) {
		cout << ">> Tracker has been deallocated." << endl;
	}

	trackers.erase(ID);
	objects.erase(ID);
	states.erase(ID);
	disappeared.erase(ID);
	
	cout << "[Done]" << endl;
}


vector<Rect> ObjectTracker::update(const Mat& frame, bool useTracker, vector<Rect> boxes) {
	// if use bounding boxes from trackers
	if (useTracker) {
		cout << "[Use tracker estimation]" << endl;

		for (const auto& e : trackers) {
			Rect2d roi;
			bool ret = e.second->update(frame, roi);

			if (ret) {
				boxes.push_back(roi);
			}
		} 
	}

	// if there is no bounding box
	if (boxes.empty()) {
		cout << "Nothing detected" << endl;
		for (auto& e : disappeared) {
			e.second++;
			if (e.second > maxDisappeared) {
				deregister(e.first);
			}
		}

		return vector<Rect>();
	}

	cout << "Boxes: " << boxes.size() << endl;

	if (objects.empty()) {
		for (const auto& e : boxes) {
			registerObj(frame, e, false);
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

		printBox(objectCentroids, "objectCentroids");
		printBox(inputCentroids, "inputCentroids");

		Mat D = cdist(objectCentroids, inputCentroids);

		vector<size_t> rows;
		vector<size_t> cols;
		vector<size_t> temp;
		for (int i = 0; i < D.size().height; i++) {
				
			vector<size_t> row;
			vector<size_t> col;
			
			for (int j = 0; j < D.size().width; j++) {
				row.push_back(D.at<uint16_t>(i, j));
				col.push_back(D.at<uint16_t>(i, j));
			}

			sort(row.begin(), row.end());
			rows.push_back(row[0]);
			temp.push_back(argmin(col));
		}

		rows = argsort(rows);

		for (size_t i = 0; i < rows.size(); i++) {
			cols.push_back(temp[rows[i]]);
		}

		printVector(rows, "rows");
		printVector(cols, "cols");

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

			cout << endl << "row: " << row << " col: " << col << " ";

			if ((usedRows.find(row) != usedRows.end()) 
				|| (usedCols.find(col) != usedCols.end())) {

				cout << ">> used" << endl;
				continue;
			}

			cout << ">> unused" << endl;

			int ID = objectIDs[row];
			if (D.at<uint16_t>(row, col) < maxDistance*maxDistance) {
				cout << ">> add to used list" << endl;
				objects[ID] = inputCentroids[col];
				disappeared[ID] = 0;
				trackers[ID]->init(frame, boxes[col]);

				usedRows.insert(row);
				usedCols.insert(col);
			}
		}

		printSet(usedRows, "usedRows");
		printSet(usedCols, "usedCols");

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

		printSet(unusedRows, "unusedRows");
		printSet(unusedCols, "unusedCols");

		//if (D.size().height >= D.size().width) {
			// some objects disappeared
			for (const auto& row : unusedRows) {
				int ID = objectIDs[row];
				disappeared[ID]++;

				if (disappeared[ID] > maxDisappeared) {
					deregister(ID);
				}
			}
		//}
		//else {
			// some new objects will be registered
		if (!useTracker) {
			for (const auto& col : unusedCols) {
				registerObj(frame, boxes[col], false);
			}
		}
		//}
	}

	return boxes;
}


map<int, Point> ObjectTracker::getCentroids() {
	return objects;
}


map<int, bool> ObjectTracker::getStates() {
	return states;
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
	Mat result(A.size(), B.size(), CV_16UC1);

	for (size_t r = 0; r < A.size(); r++) {
		for (size_t c = 0; c < B.size(); c++) {
			result.at<uint16_t>(r, c) = (A[r].x - B[c].x)*(A[r].x - B[c].x)
									  + (A[r].y - B[c].y)*(A[r].y - B[c].y);
		}
	}
	return result;
}


void test(const vector<Point>& A, const vector<Point>& B) {
	Mat D = cdist(A, B);
	cout << D << endl;

	vector<size_t> rows;
	vector<size_t> cols;
	vector<size_t> temp;
	for (int i = 0; i < D.size().height; i++) {
			
		vector<size_t> row;
		vector<size_t> col;
		
		for (int j = 0; j < D.size().width; j++) {
			row.push_back(D.at<uint16_t>(i, j));
			col.push_back(D.at<uint16_t>(i, j));
		}

		sort(row.begin(), row.end());
		rows.push_back(row[0]);
		temp.push_back(argmin(col));
	}

	rows = argsort(rows);

	for (size_t i = 0; i < rows.size(); i++) {
		cols.push_back(temp[rows[i]]);
	}

	cout << endl << "rows:";
	for (const auto& e : rows)
		cout << e << " ";
	cout << endl << "cols:";
	for (const auto& e : cols)
		cout << e << " ";
	cout << endl;
}


void ObjectTracker::showInfo() {
	cout << "Tracker: " << trackers.size() << endl;
	cout << "objects: " << objects.size() << endl;
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