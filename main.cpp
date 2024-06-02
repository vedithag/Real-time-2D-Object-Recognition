#include "main.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include "csv_util.hpp"

/*
  Tejasri Kasturi & Veditha Gudapati
  02/26/2024
  Spring 2024
  CS 5330 Computer Vision

  Functions for Real-time 2-D Object Recognition
*/

using namespace cv;

// Define constants
const char TRAINING_MODE_KEY = 'N';
const std::string FEATURE_VECTOR_FILE = "feature_vectors.csv";

bool confusionMode = false;

// pre-trained network path
string filename_dnn = "/home/veditha/Desktop/prcv/Project 3/or2d-normmodel-007.onnx";

char CSV[256] = "/home/veditha/Desktop/prcv/Project 3/feature_vectors.csv";
char CSV_DNN[256] = "/home/veditha/Desktop/prcv/Project 3/feature_vectors_dnn.csv";

// Global variables for training mode
bool trainingMode = false;
std::string trainingLabel;
std::ofstream featureVectorFile;

// Function to handle key press events
void handleKeyPress(char key)
{
    if (key == TRAINING_MODE_KEY)
    {
        trainingMode = true;
        std::cout << "Training Mode Initiated. Enter label for Detected object: ";
        std::getline(std::cin, trainingLabel);                           // Read label from console
        featureVectorFile.open(FEATURE_VECTOR_FILE, std::ios_base::app); // Open file in append mode
    }
}

// Function to store feature vector and label in a file
void storeFeatureVector(const std::string &label, const std::vector<float> &featureVector)
{
    if (featureVectorFile.is_open())
    {
        // Write label to file
        featureVectorFile << label << ",";

        // Write feature vector to file
        for (float feature : featureVector)
        {
            featureVectorFile << feature << ",";
        }
        featureVectorFile << std::endl;
    }
}

// Function to threshold the input image
void thresholdImage(const cv::Mat &src, cv::Mat &dst, int thresholdValue)
{
    cv::Mat grayscale;
    cv::cvtColor(src, grayscale, cv::COLOR_BGR2GRAY); // Convert to grayscale

    dst = cv::Mat::zeros(src.rows, src.cols, CV_8U); // Initialize the output image with zeros

    // Apply thresholding: set pixels above threshold to 0 and below threshold to 255
    for (int i = 0; i < grayscale.rows; i++)
    {
        for (int j = 0; j < grayscale.cols; j++)
        {
            uchar pixelValue = grayscale.at<uchar>(i, j);
            if (pixelValue > thresholdValue)
            {
                dst.at<uchar>(i, j) = 0;
            }
            else
            {
                dst.at<uchar>(i, j) = 255;
            }
        }
    }
}

// Function to dynamically threshold the input image based on clustering
void dynamicThresholding(const cv::Mat &src, cv::Mat &dst)
{
    // Step 1: Blur the input image
    cv::Mat blurred;
    cv::GaussianBlur(src, blurred, cv::Size(5, 5), 0);

    // Step 2: Convert the blurred image to LAB color space
    cv::Mat labImage;
    cv::cvtColor(blurred, labImage, cv::COLOR_BGR2Lab);

    // Step 3: Reshape the LAB image to a single channel (flattening)
    cv::Mat reshapedImage = labImage.reshape(1, labImage.cols * labImage.rows);
    reshapedImage.convertTo(reshapedImage, CV_32F);

    // Step 4: Apply K-means clustering to find two clusters
    int clusterCount = 2;
    cv::Mat labels;
    cv::Mat centers;
    cv::kmeans(reshapedImage, clusterCount, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0), 3, cv::KMEANS_PP_CENTERS, centers);

    // Step 5: Calculate threshold value based on cluster centers
    float center1 = (centers.at<float>(0, 0) + centers.at<float>(0, 1) + centers.at<float>(0, 2)) / 3.0f;
    float center2 = (centers.at<float>(1, 0) + centers.at<float>(1, 1) + centers.at<float>(1, 2)) / 3.0f;
    int thresholdValue = static_cast<int>((0.7 * center1 + 0.3 * center2) / 2.0f);

    // Step 6: Blur the input image again
    cv::GaussianBlur(blurred, blurred, cv::Size(5, 5), 0);

    // Step 7: Convert the blurred image to HSV color space
    cv::Mat hsvImage;
    cv::cvtColor(blurred, hsvImage, cv::COLOR_BGR2HSV);

    // Step 8: Apply thresholding based on saturation value
    for (int i = 0; i < hsvImage.rows; i++)
    {
        for (int j = 0; j < hsvImage.cols; j++)
        {
            cv::Vec3b pixel = hsvImage.at<cv::Vec3b>(i, j);
            if (pixel[1] > 100)
            {
                pixel[2] = static_cast<uchar>(pixel[2] * 0.7);
                hsvImage.at<cv::Vec3b>(i, j) = pixel;
            }
        }
    }

    // Step 9: Convert the HSV image back to BGR color space
    cv::cvtColor(hsvImage, dst, cv::COLOR_HSV2BGR);

    // Step 10: Apply thresholding
    thresholdImage(dst, dst, thresholdValue);
}

// Function to perform erosion operation on binary image
void erode(const cv::Mat &src, cv::Mat &dst, const cv::Mat &kernel)
{
    dst = src.clone(); // Clone the source image

    int kRowsHalf = kernel.rows / 2;
    int kColsHalf = kernel.cols / 2;

    // Loop through each pixel in the image
    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            bool erodePixel = true;

            // Loop through the kernel
            for (int x = -kRowsHalf; x <= kRowsHalf && erodePixel; x++)
            {
                for (int y = -kColsHalf; y <= kColsHalf; y++)
                {
                    int newX = i + x;
                    int newY = j + y;

                    // Check if the kernel overlaps with the image
                    if (newX >= 0 && newX < src.rows && newY >= 0 && newY < src.cols)
                    {
                        // Check if the corresponding kernel element is 1 and the corresponding image pixel is 0
                        if (kernel.at<uchar>(x + kRowsHalf, y + kColsHalf) == 1 && src.at<uchar>(newX, newY) == 0)
                        {
                            erodePixel = false;
                            break;
                        }
                    }
                }
            }

            // Update the destination image pixel based on erosion result
            dst.at<uchar>(i, j) = erodePixel ? 255 : 0;
        }
    }
}

// Function to perform dilation operation on binary image
void dilate(const cv::Mat &src, cv::Mat &dst, const cv::Mat &kernel)
{
    dst = src.clone(); // Clone the source image

    int kRowsHalf = kernel.rows / 2;
    int kColsHalf = kernel.cols / 2;

    // Loop through each pixel in the image
    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            bool dilatePixel = false;

            // Loop through the kernel
            for (int x = -kRowsHalf; x <= kRowsHalf && !dilatePixel; x++)
            {
                for (int y = -kColsHalf; y <= kColsHalf; y++)
                {
                    int newX = i + x;
                    int newY = j + y;

                    // Check if the kernel overlaps with the image
                    if (newX >= 0 && newX < src.rows && newY >= 0 && newY < src.cols)
                    {
                        // Check if the corresponding kernel element is 1 and the corresponding image pixel is 255
                        if (kernel.at<uchar>(x + kRowsHalf, y + kColsHalf) == 1 && src.at<uchar>(newX, newY) == 255)
                        {
                            dilatePixel = true;
                            break;
                        }
                    }
                }
            }

            // Update the destination image pixel based on dilation result
            dst.at<uchar>(i, j) = dilatePixel ? 255 : 0;
        }
    }
}

// Function to apply morphological operations (erosion and dilation) on binary image
void applyMorphologicalOperations(const cv::Mat &src, cv::Mat &dst)
{
    cv::Mat kernel = cv::Mat::ones(5, 5, CV_8U); // Define a 5x5 kernel with all elements as 1

    cv::Mat eroded;
    erode(src, eroded, kernel);  // Perform erosion
    dilate(eroded, dst, kernel); // Perform dilation
}

const double MAX_DISTANCE_TO_CENTER = 100.0; // Adjust the value as needed

// Function to segment regions in binary image
void segmentRegions(const cv::Mat &src, cv::Mat &regionMap, int largestN)
{
    cv::Mat labeledImage;
    cv::Mat stats;
    cv::Mat centroids;

    // Perform connected component analysis to label connected regions
    cv::connectedComponentsWithStats(src, labeledImage, stats, centroids);

    regionMap = cv::Mat::zeros(src.size(), CV_8UC3); // Initialize the region map with zeros

    std::vector<std::pair<int, int>> regionAreasIndices;

    // Iterate through the labeled regions
    for (int i = 1; i < stats.rows; ++i)
    {
        regionAreasIndices.emplace_back(stats.at<int>(i, cv::CC_STAT_AREA), i); // Store region area and index
    }

    // Sort regions based on area in descending order
    std::sort(regionAreasIndices.rbegin(), regionAreasIndices.rend());

    cv::RNG rng; // Random number generator

    // Iterate through the largestN regions
    for (int i = 0; i < std::min(largestN, static_cast<int>(regionAreasIndices.size())); ++i)
    {
        int index = regionAreasIndices[i].second;
        cv::Vec3b color(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)); // Random color
        cv::Mat mask = (labeledImage == index);                                         // Create mask for the region
        regionMap.setTo(color, mask);                                                   // Set pixels of the region to the color
    }
}

// Function to extract the features from the detected object
int extractFeatures(Mat input, Mat &output, vector<float> &extractedFeatures, int idOfRegion)
{
    output = input.clone(); // Clone the input image to preserve the original

    Mat convertedToGray;
    cvtColor(input, convertedToGray, COLOR_BGR2GRAY); // Convert input image to grayscale

    vector<vector<Point>> detectedContours;
    vector<Vec4i> contourHierarchy;
    findContours(convertedToGray, detectedContours, contourHierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE); // Find contours in the grayscale image

    // Sort contours based on area in descending order
    sort(detectedContours.begin(), detectedContours.end(), [](const vector<Point> &a, const vector<Point> &b)
         {
             return contourArea(a) > contourArea(b); // Sorting based on contour area
         });

    for (size_t idx = 0; idx < detectedContours.size(); idx++)
    {
        if (contourArea(detectedContours[idx]) > 0)
        {                                                                      // Check if the contour area is valid
            Moments calculatedMoments = moments(detectedContours[idx], false); // Calculate moments for the contour

            double huMoments[7];
            HuMoments(calculatedMoments, huMoments); // Calculate Hu moments for shape descriptor

            for (int j = 0; j < 7; j++)
            {
                huMoments[j] = -1 * copysign(1.0, huMoments[j]) * log10(abs(huMoments[j])); // Normalize and transform Hu moments
                extractedFeatures.push_back(huMoments[j]);                                  // Store Hu moments in the feature vector
            }

            RotatedRect boundingRect = minAreaRect(detectedContours[idx]);                                                      // Get the minimum bounding rectangle for the contour
            Point2f centerOfMass(calculatedMoments.m10 / calculatedMoments.m00, calculatedMoments.m01 / calculatedMoments.m00); // Calculate the center of mass
            float maxWidth = max(boundingRect.size.width, boundingRect.size.height);                                            // Calculate maximum width of the bounding rectangle
            float minHeight = min(boundingRect.size.width, boundingRect.size.height);                                           // Calculate minimum height of the bounding rectangle
            double rotationAngle = boundingRect.angle;                                                                          // Get the rotation angle of the bounding rectangle
            if (boundingRect.size.width < boundingRect.size.height)
                rotationAngle += 90.0;                                             // Adjust rotation angle if necessary
            float aspectRatio = maxWidth / minHeight;                              // Calculate aspect ratio of the bounding rectangle
            float fillPercentage = calculatedMoments.m00 / (maxWidth * minHeight); // Calculate fill percentage of the region

            extractedFeatures.push_back(aspectRatio);    // Store aspect ratio in the feature vector
            extractedFeatures.push_back(fillPercentage); // Store fill percentage in the feature vector

            // Drawing the bounding rectangle
            Point2f rectangleVertices[4];
            boundingRect.points(rectangleVertices); // Get the vertices of the bounding rectangle
            for (int vertex = 0; vertex < 4; vertex++)
            {
                line(output, rectangleVertices[vertex], rectangleVertices[(vertex + 1) % 4], Scalar(0, 255, 0), 2); // Draw the bounding rectangle
            }

            // Drawing the orientation line
            double radianAngle = rotationAngle * CV_PI / 180.0;
            Point2f endPoint(centerOfMass.x + 0.5 * maxWidth * cos(radianAngle), centerOfMass.y + 0.5 * maxWidth * sin(radianAngle)); // Calculate endpoint of the orientation line
            line(output, centerOfMass, endPoint, Scalar(255, 0, 0), 2);                                                               // Draw the orientation line

            // Displaying the text
            stringstream textToDisplay;
            textToDisplay << "Region " << idx + 1 << ": Aspect Ratio=" << aspectRatio << ", Fill=" << fillPercentage;   // Text to display
            Point positionForText(centerOfMass.x - 50, centerOfMass.y + 20);                                            // Position for displaying text
            putText(output, textToDisplay.str(), positionForText, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1); // Put text on the image
        }
    }

    return 0; // Return success
}

// Calculate the standard deviation for each entry of the features in the database (for scale)
int standardDeviation(std::vector<std::vector<float>> &data, std::vector<float> &deviations)
{
    // Initialize vector to store standard deviations
    deviations = std::vector<float>(data[0].size(), 0.0);

    // Calculate the mean for each feature
    std::vector<float> means(data[0].size(), 0.0);
    for (const auto &entry : data)
    {
        for (size_t j = 0; j < entry.size(); ++j)
        {
            means[j] += entry[j] / data.size(); // Accumulate sum of values for each feature
        }
    }

    // Calculate the sum of squared differences
    std::vector<float> sumSquare(data[0].size(), 0.0);
    for (const auto &entry : data)
    {
        for (size_t j = 0; j < entry.size(); ++j)
        {
            sumSquare[j] += std::pow(entry[j] - means[j], 2); // Accumulate sum of squared differences
        }
    }

    // Calculate the standard deviation for each feature
    for (size_t i = 0; i < sumSquare.size(); ++i)
    {
        deviations[i] = std::sqrt(sumSquare[i] / (data.size() - 1)); // Calculate standard deviation using sum of squared differences
    }

    return 0; // Return success
}

// Euclidean distance function
double euclideanDistance(const std::vector<float> &point1, const std::vector<float> &point2)
{
    double distance = 0.0;

    // Iterate through each dimension of the feature vectors and calculate the squared difference
    for (size_t i = 0; i < point1.size(); ++i)
    {
        distance += std::pow(point1[i] - point2[i], 2); // Calculate squared difference for each dimension
    }

    return std::sqrt(distance); // Return the square root of the sum of squared differences
}

// Function to calculate the scaled Euclidean Distance between two feature vectors
// This distance is calculated as the absolute difference between corresponding feature values divided by the standard deviation
float scaledEuclideanDis(const std::vector<float> &feature1, const std::vector<float> &feature2, const std::vector<float> &deviations)
{
    float distance = 0.0;

    // Iterate through each dimension of the feature vectors and calculate the scaled absolute difference
    for (int i = 0; i < feature1.size(); i++)
    {
        distance += std::abs(feature1[i] - feature2[i]) / deviations[i]; // Calculate scaled absolute difference for each dimension
    }

    return distance; // Return the sum of scaled absolute differences
}

// Function for classifying images based on input features
string classifyingImages(vector<float> &features)
{

    // Define the file path containing the dataset of feature vectors
    char fileName[256] = "/home/veditha/Desktop/prcv/Project 3/feature_vectors.csv";

    // Initialize vectors to store labels and feature vectors extracted from the dataset
    std::vector<char *> labels;
    std::vector<std::vector<float>> nfeatures;

    // Read image data from the CSV file into the defined vectors
    read_image_data_csv(fileName, labels, nfeatures, 0);

    // Initialize the minimum distance to positive infinity and an empty string for the minimum label found
    double min_dist = std::numeric_limits<double>::infinity();
    string min_label;

    // Calculate standard deviation for each feature dimension in the dataset
    std::vector<float> deviations;
    standardDeviation(nfeatures, deviations);

    // Iterate through each feature vector in the dataset
    for (int i = 0; i < nfeatures.size(); i++)
    {

        // Calculate the scaled Euclidean distance between the input features and the features in the dataset
        double dist = scaledEuclideanDis(nfeatures[i], features, deviations);

        // Update the minimum distance and corresponding label if the calculated distance is smaller
        if (dist < min_dist)
        {
            min_dist = dist;
            min_label = labels[i];
        }
    }
    return min_label; // Return the label corresponding to the feature vector with the smallest distance
}

int getEmbedding(cv::Mat &src, cv::Mat &embedding, cv::Rect &bbox, cv::dnn::Net &net, int debug)
{
    const int ORNet_size = 128;
    cv::Mat padImg;
    cv::Mat blob;

    cv::Mat roiImg = src(bbox);
    int top = bbox.height > 128 ? 10 : (128 - bbox.height) / 2 + 10;
    int left = bbox.width > 128 ? 10 : (128 - bbox.width) / 2 + 10;
    int bottom = top;
    int right = left;

    cv::copyMakeBorder(roiImg, padImg, top, bottom, left, right, cv::BORDER_CONSTANT, 0);
    cv::resize(padImg, padImg, cv::Size(128, 128));

    cv::dnn::blobFromImage(src,                              // input image
                           blob,                             // output array
                           (1.0 / 255.0) / 0.5,              // scale factor
                           cv::Size(ORNet_size, ORNet_size), // resize the image to this
                           128,                              // subtract mean prior to scaling
                           false,                            // input is a single channel image
                           true,                             // center crop after scaling short side to size
                           CV_32F);                          // output depth/type

    net.setInput(blob);
    embedding = net.forward("onnx_node!/fc1/Gemm");

    if (debug)
    {
        cv::imshow("pad image", padImg);
        std::cout << embedding << std::endl;
        cv::waitKey(0);
    }

    return (0);
}

// Function to calculate the Euclidean distance between two feature vectors
float euclideanDist(const vector<float> &f1, const vector<float> &f2)
{
    float distanceSquared = 0.0;

    // Iterate through each dimension of the feature vectors and calculate the squared difference
    for (size_t i = 0; i < f1.size(); ++i)
        distanceSquared += pow((f1[i] - f2[i]), 2); // Calculate squared difference for each dimension

    return sqrt(distanceSquared); // Return the square root of the sum of squared differences
}

// Function to calculate the cosine distance between two feature vectors
float cosDistance(const vector<float> &v1, const vector<float> &v2)
{
    float dotProduct = 0.0;
    float normV1 = 0.0;
    float normV2 = 0.0;

    // Iterate through each dimension of the feature vectors and compute dot product and norms
    for (size_t i = 0; i < v1.size(); ++i)
    {
        dotProduct += v1[i] * v2[i]; // Calculate dot product
        normV1 += pow(v1[i], 2);     // Calculate sum of squares for vector 1
        normV2 += pow(v2[i], 2);     // Calculate sum of squares for vector 2
    }

    normV1 = sqrt(normV1); // Calculate square root of sum of squares for vector 1
    normV2 = sqrt(normV2); // Calculate square root of sum of squares for vector 2

    // Check for division by zero and return 0 if either norm is 0
    if (normV1 == 0.0 || normV2 == 0.0)
    {
        return 0.0; // Prevent division by zero
    }
    else
    {
        return dotProduct / (normV1 * normV2); // Calculate cosine distance
    }
}

// Function to calculate the sum of squared differences between two feature vectors
double sumSquaredDifference(const vector<float> &a, const vector<float> &b)
{
    double sumSquaredDiff = 0.0;

    // Iterate through each dimension of the feature vectors and calculate the squared difference
    for (size_t i = 0; i < a.size(); i++)
    {
        sumSquaredDiff += pow((a[i] - b[i]), 2); // Calculate squared difference for each dimension
    }
    return sumSquaredDiff; // Return the sum of squared differences
}

// Function to classify feature vectors using a deep neural network (DNN) based approach
string classifyDNN(vector<float> &features)
{
    // Define the file path containing the dataset of DNN-based feature vectors
    char fileName[256] = "/home/kasturi/PRCV/Project_3/feature_vectors_dnn.csv";

    // Initialize vectors to store labels and feature vectors extracted from the DNN-based dataset
    std::vector<char *> labels;
    std::vector<std::vector<float>> nfeatures;

    // Read image data from the CSV file into the defined vectors
    read_image_data_csv(fileName, labels, nfeatures, 0);

    // Initialize the minimum distance to positive infinity and an empty string for the minimum label found
    double min_dist = std::numeric_limits<double>::infinity();
    string min_label;

    // Iterate through each feature vector in the dataset
    for (int i = 0; i < nfeatures.size(); i++)
    {
        // Calculate the cosine distance between the input features and the features in the dataset
        double dist = cosDistance(nfeatures[i], features);

        // Update the minimum distance and corresponding label if the calculated distance is smaller
        if (dist < min_dist)
        {
            min_dist = dist;
            min_label = labels[i];
            // Print debugging information: label and distance
            cout << "Label: " << min_label << endl;
            cout << "Dist: " << dist << endl;
        }
    }

    return min_label; // Return the label corresponding to the feature vector with the smallest distance
}

/* The function converts a Matrix into a vector<float> type.
 * This function is used to convert the embedding matrix from a deep neural network (DNN) into a vector<float> type
 * to append it to a CSV file.
 *
 * Parameters:
 * - mat: The input matrix to be converted into a vector<float>.
 *
 * Returns:
 * A vector<float> containing the elements of the input matrix.
 */
std::vector<float> matToVector(const cv::Mat &mat)
{
    std::vector<float> vec;
    // Ensure that the input matrix is not empty
    if (!mat.empty())
    {
        // Iterate through the matrix rows
        for (int i = 0; i < mat.rows; ++i)
        {
            // Iterate through the matrix columns
            for (int j = 0; j < mat.cols; ++j)
            {
                // Push the element into the vector
                vec.push_back(mat.at<float>(i, j));
            }
        }
    }
    return vec; // Return the resulting vector<float>
}

int main()
{
    // Initialize vectors to store features and embedding features
    vector<float> feature;
    vector<float> embeddingFeature;

    // Read the pre-trained deep network
    cv::dnn::Net net = cv::dnn::readNet(filename_dnn);

    // Define a map to associate object labels with integer indices
    std::map<std::string, int> mpp;
    mpp["box"] = 1;
    mpp["hammer"] = 2;
    mpp["dog"] = 3;
    mpp["mouse"] = 4;
    mpp["book"] = 5;

    // Initialize a confusion matrix to evaluate classification performance
    vector<vector<int>> confusionMat(5, vector<int>(5, 0));

    // Open the camera for capturing video frames
    cv::VideoCapture cap(0);

    // Check if the camera is opened successfully
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open camera." << std::endl;
        return -1;
    }

    // Initialize variables for feature extraction and region processing
    int largestN = 3;
    int regionID = 1;
    char key = 0;

    // Main loop to process video frames
    while (true)
    {
        cv::Mat frame;
        cap >> frame;

        // Check if the frame is empty
        if (frame.empty())
        {
            std::cerr << "Error: Couldn't capture a frame." << std::endl;
            break;
        }

        // Declare matrices for intermediate processing steps
        cv::Mat thresholdedFrame;
        cv::Mat morphological;
        cv::Mat regionMap;
        cv::Mat featuresDetected;

        // Apply dynamic thresholding to the captured frame
        dynamicThresholding(frame, thresholdedFrame);

        // Apply morphological operations to enhance the thresholded image
        applyMorphologicalOperations(thresholdedFrame, morphological);

        // Segment regions in the morphological image
        segmentRegions(morphological, regionMap, largestN);

        // Clone the original frame to display detected features
        featuresDetected = frame.clone();

        // Clear the feature vector for the current region
        feature.clear();

        // Extract features from the region map
        extractFeatures(regionMap, featuresDetected, feature, regionID);

        // Wait for key press to trigger specific actions
        key = cv::waitKey(30);
        if (key == 'N' || key == 'n')
        {
            // Enter training mode to label the detected object
            char label[20];
            std::cout << "Training Mode Initiated. Enter label for the detected object: ";
            std::cin >> label;

            // Append the detected features and label to the CSV file
            append_image_data_csv(CSV, label, feature, 0);
        }
        else if (key == 'r')
        {
            // Classify the detected object using the KNN algorithm
            string temp = classifyingImages(feature);
            cout << temp << endl;
        }
        else if (key == 'e')
        {
            // Classify the detected object using the pre-trained DNN model
            string temp = classifyDNN(feature);
            cout << temp << endl;
        }
        else if (key == 'l' || key == 'L')
        {
            // Training mode initiated for DNN classification
            char labelDNN[20];
            std::cout << "Training Mode Initiated. Enter label for the detected object: ";
            std::cin >> labelDNN;

            // Get the embedding for the detected object
            Rect bbox(0, 0, thresholdedFrame.cols, thresholdedFrame.rows);
            Mat embedding;
            getEmbedding(thresholdedFrame, embedding, bbox, net, 1); // change the 1 to a 0 to turn off debugging
            embeddingFeature = matToVector(embedding);

            // Append the detected features and label to the CSV file for DNN training
            append_image_data_csv(CSV_DNN, labelDNN, embeddingFeature, 0);
        }
        else if (key == 'C' || key == 'c')
        {
            // Confusion matrix mode initiated
            std::cout << "Confusion matrix :" << std::endl;

            // Prompt user to enter the actual object label for confusion matrix
            string confLabel;
            cout << "Enter the object label: " << endl;
            cin >> confLabel;

            // Determine the actual label index from the label map
            int actualLabel = mpp[confLabel] - 1;

            // Perform object classification to get the predicted label
            string rand = classifyingImages(feature);
            cout << rand << endl;
            int predLabel = mpp[classifyingImages(feature)] - 1;

            // Update confusion matrix based on actual and predicted labels
            confusionMat[actualLabel][predLabel]++;

            // Display the confusion matrix
            cout << "Matrix elements:" << endl;
            for (const auto &row : confusionMat)
            {
                for (const auto &element : row)
                {
                    cout << element << " ";
                }
                cout << endl;
            }
        }
        else if (key == 'd')
        {
            // DNN classification mode initiated

            // Get the embedding for the detected object
            Rect bbox(0, 0, thresholdedFrame.cols, thresholdedFrame.rows);
            Mat embedding;
            getEmbedding(thresholdedFrame, embedding, bbox, net, 1); // change the 1 to a 0 to turn off debugging
            embeddingFeature = matToVector(embedding);

            // Classify the detected object using the DNN model
            string temp2 = classifyDNN(embeddingFeature);

            // Identify the object
            cout << "The object is: " << temp2 << endl;
            cout << "Exiting DNN Mode" <<
        }

        else if (key == 27)
        { // Press ESC to exit
            break;
        }

        // Display the original image, thresholded image, morphological image,
        // region map, and features detected images for visual inspection
        cv::imshow("Original Image", frame);
        cv::imshow("Thresholded Image", thresholdedFrame);
        cv::imshow("Morphological Image", morphological);
        cv::imshow("Region Map", regionMap);
        cv::imshow("Features Detected", featuresDetected);

        // Store the feature vector if not empty and write to file
        if (!feature.empty() && !trainingLabel.empty() && featureVectorFile.is_open())
        {
            storeFeatureVector(trainingLabel, feature);
            std::cout << "Feature vector stored for label: " << trainingLabel << std::endl;
            featureVectorFile.close(); // Close the file after storing the feature vector
        }

        // Increment region ID and reset if exceeds largestN
        regionID++;
        if (regionID > largestN)
        {
            regionID = 1;
        }
    }

    // Release the camera and close all OpenCV windows before exiting
    cap.release();
    cv::destroyAllWindows();

    return 0; // Exit the program with status code 0
}

