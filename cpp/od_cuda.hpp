#include <iostream>
#include <queue>
#include <iterator>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


constexpr float CONFIDENCE_THRESHOLD = 0.25;
constexpr float NMS_THRESHOLD = 0.4;

std::vector<cv::Mat> run_inference(cv::Mat frame);
void parse_yolo_output( cv::Mat frame,std::vector<cv::Mat> detections);
void init_model();
void od_gen_debug_image(cv::Mat frame, std::string stats);
void reset_all_vectors();
cv::Point compute2DPolygonCentroid(cv::Point* vertices, int vertexCount);
void set_up_bed_center(int *bed_mask);
int do_proximity_detection();
void run_destructor();
cv::Mat get_od_frame();