#include <iostream>
#include <string>
using namespace std;

#include <opencv2\opencv.hpp>

#include "../clcnst/clcnst.h"

float sigma;
string ifname, ofname, isex;

int main(int argc, char** argv) {
	// Load input image
	cout << "[MooreAlgorithm] input file name to load: ";
	cin >> ifname;

	cv::Mat img = cv::imread(ifname, CV_LOAD_IMAGE_COLOR);
	if(img.empty()) {
		cout << "Failed to load file \"" << ifname << "\"." << endl;
		return -1;
	}
	int width = img.cols;
	int height = img.rows;
	int channel = img.channels();
	img.convertTo(img, CV_32FC3, 1.0 / 255.0);
	
	// Apply Gaussian filter
	cout << "[MooreAlgorithm] input sigma value for Gaussian: ";
	cin >> sigma;
	sigma = sigma * max(width, height);

	cv::Mat gauss;
	cv::GaussianBlur(img, gauss, cv::Size(0, 0), sigma);

	// Additional process for extended Moore
	cout << "[MooreAlgorithm] use extended algorithm? (y/n): ";
	cin >> isex;
	if(isex == "y") {
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
	cv::Mat out;
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
	cout << "[MooreAlgorithm] save as: ";
	cin >> ofname;
	out.convertTo(out, CV_8UC3, 255.0);
	cv::imwrite(ofname, out);
}