#include <iostream>
#include <string>
using namespace std;

#include <opencv2/opencv.hpp>

#include "../clcnst/clcnst.h"

int ns;
float sigma, scale;
string ifname, ofname, isp;

int main(int argc, char** argv) {
	// Load input image
	cout << "[RahmanAlgorithm] input file name to load: ";
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

	// Obtain parameters by keyboard interaction
	cout << "[RahmanAlgorithm] you want to specify parameters? (y/n): ";
	cin >> isp;
	if(isp == "y") {
		cout << "  sigma = ";
		cin >> sigma;
		cout << "  number of sigmas = ";
		cin >> ns;
		cout << "  scales for sigmas = ";
		cin >> scale;		
	} else {
		sigma = 1.0f;
		ns = 3; 
		scale = 0.16f;
	}

	vector<float> sigmas = vector<float>(ns);
	sigmas[0] = sigma * (float)max(height, width);
	for(int i=1; i<ns; i++) sigmas[i] = sigmas[i-1] * scale;

	// Accumulate multiscale results of Moore's algorithm
	cv::Mat out, tmp, gauss;
	double weight = 0.0;
	out = cv::Mat(height, width, CV_32FC3);
	for(int i=0; i<ns; i++) {
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
	cout << "[RahmanAlgorithm] save as: ";
	cin >> ofname;
	out.convertTo(out, CV_8UC3, 255.0);
	cv::imwrite(ofname, out);
}
