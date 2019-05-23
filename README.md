## class ObjectTracker
### Purpose:
- used for object tracking and counting;
### Dependencies:
- OpenCV 3.3+
### Usage:
- constructor: **ObjectTracker**(*int* maxFrame = 30) - Create a tracker that will be removed if object disappears 
in 30 consecutive frames (default).
- update tracker: *vector<cv::Rect>* **update**(*const cv::Mat&* frame, *bool* usedTracker = true, *vector<cv::Rect>* boxes = vector<cv::Rect>()) - Update 
tracker with boxes detected by Object Detector or receive new object's locations estimated by tracker.
- get centroids of objects: *map<int, cv::Point>* **getCentroids**().
- get states of objects: *map<int, bool>* **getStates**().
