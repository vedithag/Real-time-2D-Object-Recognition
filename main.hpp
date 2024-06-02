#ifndef MAIN_HPP
#define MAIN_HPP

#include <opencv2/opencv.hpp> // Include OpenCV library
#include <vector> // Include vector container for storing data

using namespace cv; // Use the OpenCV namespace
using namespace std; // Use the standard namespace

// Define the structure for region features
struct region_features {
    float ratio; // Ratio feature for a region
    float percent_filled; // Percentage filled feature for a region
};

// Function declarations

// Function to threshold the input image
void thresholdImage(const cv::Mat& src, cv::Mat& dst, int thresholdValue);

// Function to dynamically threshold the input image based on clustering
void dynamicThresholding(const cv::Mat& src, cv::Mat& dst);

// Function to perform dilation operation on binary image
void dilate(const cv::Mat& src, cv::Mat& dst, const cv::Mat& kernel);

// Function to perform erosion operation on binary image
void erode(const cv::Mat& src, cv::Mat& dst, const cv::Mat& kernel);

// Function to apply morphological operations (dilation and then erosion) on binary image
void applyMorphologicalOperations(const cv::Mat& src, cv::Mat& dst);

// Function to segment regions in binary image
void segmentRegions(const cv::Mat& src, cv::Mat& regionMap, int largestN);

// Function to check if a region touches the image boundary
bool touchesBoundary(const cv::Mat& stats, int regionIndex, const cv::Size& imageSize);

// Function to compute features of regions in a binary image
int compute_features(Mat src, Mat& dst, vector<float>& features);

// Function to calculate the standard deviation for each entry of the features in the database
int standardDeviation(std::vector<std::vector<float>>& data, std::vector<float>& deviations);

// Function for image classification based on extracted features
string classifyimgs(vector<float>& features);

// End of function declarations

#endif // MAIN_HPP

