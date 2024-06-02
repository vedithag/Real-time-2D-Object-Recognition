# Real-time-2D-Object-Recognition

## Introduction
This project aims to develop a real-time 2D object recognition system capable of identifying objects placed on a white surface from an overhead view. The system integrates advanced computer vision algorithms and methodologies to achieve accurate object classification. The process begins by converting color images into binary representations using thresholding techniques, allowing the foreground objects to be separated from the background. This initial segmentation is further refined through morphological processes, enhancing the precision of object identification.

The segmented binary image is then divided into sections, enabling the detection and labeling of connected components. This prepares the image for feature extraction, where relevant characteristics of each detected region are calculated. These characteristics are crucial for the classification process, serving as the basis for distinguishing and categorizing objects.

The system also incorporates a training mode, allowing it to learn and categorize objects in real-time based on user-provided labels. The performance of the classification system is evaluated using metrics such as accuracy, precision, and recall, providing insights for algorithmic improvements. Additionally, the project explores a secondary classification method using deep neural networks (DNNs), leveraging their ability to autonomously build hierarchical representations of features from raw data.

## Features
Image Thresholding: Converts color images to binary images using dynamic thresholding techniques.
Morphological Operations: Enhances binary image segmentation through erosion and dilation.
Region Segmentation: Identifies and labels connected regions within binary images.
Feature Extraction: Calculates features such as Hu moments and bounding rectangles for each region.
Training Mode: Allows the system to learn and categorize objects based on user-provided labels.
Real-time Classification: Uses k-Nearest Neighbors (k-NN) and deep neural networks (DNNs) for real-time object classification.
Performance Evaluation: Assesses the system’s performance using accuracy, precision, recall, and confusion matrices.

## Tasks
### Task 1: Threshold the Input Video
Function: thresholdImage
Converts color images to grayscale and then to binary by comparing pixel intensity values to a predefined threshold.
Dynamic Thresholding: Adjusts threshold values based on clustering and saturation values using Gaussian blurring, color space conversion, and K-means clustering.

### Task 2: Clean Up the Binary Image
Function: applyMorphologicalOperations
Utilizes erosion and dilation to refine binary image segmentation by removing noise and enhancing object boundaries.

### Task 3: Segment the Image into Regions
Function: segmentRegions
Performs connected component analysis to label and visualize distinct regions within binary images using contours and bounding boxes.

### Task 4: Compute Features for Each Major Region
Function: extractFeatures
Extracts features from segmented regions, including contour detection, spatial moments, Hu moments, and bounding rectangles.

### Task 5: Collect Training Data
Training Mode Activation: Triggered by pressing 'N' key.
Feature Vector Storage: Saves extracted features and user-provided labels in a CSV file for training the classification algorithm.

### Task 6: Classify New Images
Classification: Initiated by pressing 'r' key.
Uses Euclidean distance to compare extracted features with those in the training dataset, assigning the nearest label to the object.

### Task 7: Evaluate the Performance of Your System
Confusion Matrix: Used to evaluate classification performance by comparing predicted labels with actual labels, identifying true/false positives and negatives.

### Task 8: Capture A Demo of Your System Working
A demo video showcasing the system in action can be found here.

### Task 9: Implement A Second Classification Method
Deep Neural Networks (DNNs): Explored for their capability to learn hierarchical representations from raw data, providing flexibility and robustness over k-NN.
