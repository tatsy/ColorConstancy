#include <iostream>
#include <cmath>
#include <vector>
#include <string>
using namespace std;

#include <opencv2\opencv.hpp>

#include "../clcnst/clcnst.h"

string ifname, ofname, isp;
float lower, upper, threshold;

int main(int argc, char** argv) {
	// Load input image
	cout << "[HomomorphicFilter] input file name to load: ";
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

	// Obtain parameters from command line arguments
	cout << "[HomomorphicFilter] you want to specify parameters? (y/n): ";
	cin >> isp;
	if(isp == "y") {
		cout << "  scale for  low frequency (default = 0.5): ";
		cin >> lower;
		cout << "  scale for high frequency (default = 2.0): ";
		cin >> upper;
		cout << "  threshold value for frequency domain (default = 7.5):";
		cin >> threshold;
	} else {
		lower = 0.5f;
		upper = 2.0f;
		threshold = 7.5f;
	}

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
	cout << "[HomomorphicFilter] save as: ";
	cin >> ofname;
	out.convertTo(out, CV_8UC3, 255.0);
	cv::imwrite(ofname, out);
}
