#include <iostream>
using namespace std;

#include <opencv2\opencv.hpp>

#include "../clcnst/clcnst.h"

int main(int argc, char** argv) {

	// Check input command arguments
	if(argc < 3) {
		cout << "usage: MooreAlgorithm.exe [input image] [output image] [sigma for Gaussian] [option]" << endl;
		cout << "Options: " << endl;
		cout << "  -e : Use extended Moore's algorithm" << endl;
		return -1;
	}

	cv::Mat img = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
	if(img.empty()) {
		cout << "Failed to load file \"" << argv[1] << "\"." << endl;
		return -1;
	}
	int width = img.cols;
	int height = img.rows;
	int channel = img.channels();
	img.convertTo(img, CV_32FC3, 1.0 / 255.0);
	cout << "Image file \"" << argv[1] << "\" has been loaded." << endl;

	cv::Mat out, gauss;

	// Apply Gaussian filter
	float sigma = argc > 3 ? (float)atof(argv[3]) : 1.0f;
	sigma *= (float)max(width, height);

	cv::GaussianBlur(img, gauss, cv::Size(0, 0), sigma);

	if(argc > 4 && !strcmp(argv[4], "-e")) {
		cv::Mat gray;
		cv::cvtColor(img, gray, CV_BGR2GRAY);

		cv::Mat edge = cv::Mat::zeros(height, width, CV_32FC1);
		for(int y=1; y<height-1; y++) {
			for(int x=1; x<width-1; x++) {
				float dx = (gray.at<float>(y, x+1) - gray.at<float>(y, x-1)) / 2.0f;
				float dy = (gray.at<float>(y+1, x) - gray.at<float>(y-1, x)) / 2.0f;
				edge.at<float>(y, x) = sqrt(dx * dx + dy * dy);
			}
		}

		cv::GaussianBlur(edge, edge, cv::Size(0, 0), sigma);
		cv::namedWindow("Edge");
		cv::imshow("Edge", edge);
		
		for(int y=0; y<height; y++) {
			for(int x=0; x<width; x++) {
				for(int c=0; c<channel; c++) {
					gauss.at<float>(y, x*channel+c) *= edge.at<float>(y, x);
				}
			}
		}
	}

	// Subtraction
	cv::subtract(img, gauss, out);

	// Offset reflectance
	out.convertTo(out, CV_32FC3, 1.0, -1.0);

	// Normalization
	clcnst::normalize(out, out, 0.0f, 1.0f);

	// Display result
	cv::namedWindow("Input");
	cv::namedWindow("Output");
	cv::imshow("Input", img);
	cv::imshow("Output", out);
	cv::waitKey(0);
	cv::destroyAllWindows();

	// Save output image
	out.convertTo(out, CV_8UC3, 255.0);
	cv::imwrite(argv[2], out);
}