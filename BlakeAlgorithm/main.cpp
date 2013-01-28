#include <iostream>
#include <string>
using namespace std;

#include <opencv2\opencv.hpp>

#include "../clcnst/clcnst.h"

float threshold;
string ifname, ofname;

int main(int argc, char** argv) {
	// Load input file
	cout << "[BlakeAlgorithm] input file name: ";
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

	// Obtain threshold value by keyborad interaction
	cout << "[BlakeAlgorithm] input threshold value (default = 0.10): ";
	cin >> threshold;

	// Compute logarithm of input image
	cv::Mat out;
	clcnst::logarithm(img, out);

	// Laplacian filter divided by thresholding
	cv::Mat laplace = cv::Mat::zeros(height, width, CV_32FC3);
	for(int c=0; c<channel; c++) {
		// Compute gradient and thresholding
		cv::Mat grad = cv::Mat::zeros(height, width, CV_32FC2);
		for(int y=0; y<height-1; y++) {
			for(int x=0; x<width-1; x++) {
				float dx = out.at<float>(y, (x+1)*channel+c) - out.at<float>(y, x*channel+c);
				float dy = out.at<float>(y+1, x*channel+c) - out.at<float>(y, x*channel+c);
				if(fabs(dx) > threshold) {
					grad.at<float>(y, x*2+0) = dx;
				}

				if(fabs(dy) > threshold) {
					grad.at<float>(y, x*2+1) = dy;
				}
			}
		}

		// Compute gradient again
		for(int y=1; y<height; y++) {
			for(int x=1; x<width; x++) {
				float ddx = grad.at<float>(y, x*2+0) - grad.at<float>(y, (x-1)*2+0);
				float ddy = grad.at<float>(y, x*2+1) - grad.at<float>(y-1, x*2+1);
				laplace.at<float>(y, x*channel+c) = ddx + ddy;
			}
		}
	}

	// Gauss Seidel method
	clcnst::gauss_seidel(out, laplace, 20);

	// Normalization
	clcnst::normalize(out, out);

	// Compute exponential
	clcnst::exponential(out, out);

	// Display result
	cv::namedWindow("Input");
	cv::namedWindow("Output");
	cv::imshow("Input", img);
	cv::imshow("Output", out);
	cv::waitKey(0);
	cv::destroyAllWindows();

	// Save output
	cout << "[BlakeAlgorithm] save as: ";
	cin >> ofname;
	out.convertTo(out, CV_8UC3, 255.0);
	cv::imwrite(ofname, out);

}