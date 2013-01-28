#include <iostream>
#include <cmath>
#include <vector>
using namespace std;

#include <opencv2\opencv.hpp>

#include "../clcnst/clcnst.h"

int main(int argc, char** argv) {
	// Check input command arguments
	if(argc < 3) {
		cout << "usage: MooreAlgorithm.exe [input image] [output image]" << endl;
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

	// Obtain parameters from command line arguments
	float lower = argc > 4 ? (float)atof(argv[3]) : 0.5;
	float upper = argc > 4 ? (float)atof(argv[4]) : 1.0;
	float threshold = argc > 5 ? (float)atof(argv[5]) : 10.0;

	// Perform DFT, high emphasis filter and IDFT
	vector<cv::Mat> chs, spc(channel, cv::Mat(height, width, CV_32FC1));
	cv::split(img, chs);

	for(int c=0; c<channel; c++) {
		cv::dct(chs[c], spc[c]);
		clcnst::hef(spc[c], spc[c], lower, upper, threshold);
		cv::idct(spc[c], chs[c]);
	}
	cv::Mat out;
	cv::merge(chs, out);

	// Display result
	cv::namedWindow("Input");
	cv::namedWindow("Output");
	cv::imshow("Input", img);
	cv::imshow("Output", out);
	cv::waitKey(0);
	cv::destroyAllWindows();

	// Save result
	cv::imwrite(argv[2], out);
}
