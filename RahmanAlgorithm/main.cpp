#include <iostream>
using namespace std;

#include <opencv2\opencv.hpp>

#include "../clcnst/clcnst.h"

int main(int argc, char** argv) {

	// Check input command arguments
	if(argc < 5) {
		cout << "usage: RahmanAlgorithm.exe [input image] [output image] [sigma] [number of sigmas] [scale]" << endl;
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

	cv::Mat out, tmp, gauss;

	float sigma = argc >= 5 ? (float)atof(argv[3]) : 1.0f;
	float ns = argc >= 5 ? atoi(argv[4]) : 3; 
	float scale = argc >= 5 ? (float) atof(argv[5]) : 0.16f;

	vector<float> sigmas = vector<float>(ns);
	sigmas[0] = sigma * (float)max(height, width);
	for(int i=1; i<=ns; i++) sigmas[i] = sigmas[i-1] * scale;

	double weight = 0.0;
	out = cv::Mat(height, width, CV_32FC3);
	for(int i=0; i<ns; i++) {
		cout << sigmas[i] << endl;
		// Apply Gaussian filter
		cv::GaussianBlur(img, gauss, cv::Size(0, 0), sigmas[i]);

		// Subtraction
		cv::subtract(img, gauss, tmp);

		// Offset reflectance
		tmp.convertTo(tmp, CV_32FC3, 1.0, -1.0);

		// Normalization
		clcnst::normalize(tmp, tmp, 0.0f, 1.0f);

		// Accumulate
		cv::scaleAdd(tmp, 1.0 / (i+1), out, out);
		weight += 1.0 / (i+1);
	}
	out.convertTo(out, CV_32FC3, 1.0 / weight);

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