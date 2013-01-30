#include <iostream>
#include <string>
using namespace std;

#include <opencv2/opencv.hpp>

#include "../clcnst/clcnst.h"

int ns;
float sigma, scale;
string ifname, ofname;

void hef_faugeras(cv::Mat& input, cv::Mat& output) {
	cv::Mat* i_ptr = &input;
	int width = i_ptr->cols;
	int height = i_ptr->rows;
	int channel = i_ptr->channels();

	cv::Mat* o_ptr = &output;
	if(i_ptr != o_ptr) {
		*o_ptr = cv::Mat(height, width, CV_MAKETYPE(CV_32F, channel));
	}

	for(int y=0; y<height; y++) {
		for(int x=0; x<width; x++) {
			float r = sqrt((float)(x*x + y*y));
			double coeff = 1.5f - exp(- r / 5.0f);
			for(int c=0; c<channel; c++) {
				o_ptr->at<float>(y, x*channel+c) = coeff * i_ptr->at<float>(y, x*channel+c);
			}
		}
	}
}

int main(int argc, char** argv) {
	// Load input image
	cout << "[FaugerasAlgorithm] input file name to load: ";
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
	
	// Convert color space 
	cv::Mat hvs;
	cv::cvtColor(img, hvs, CV_BGR2Lab);

	// Homomophic filtering
	vector<cv::Mat> chs, spc(channel, cv::Mat(height, width, CV_32FC1));
	cv::split(hvs, chs);

	for(int c=1; c<channel; c++) {
		cv::dct(chs[c], spc[c]);
		hef_faugeras(spc[c], spc[c]);
		cv::idct(spc[c], chs[c]);
	}
	cv::Mat out;
	cv::merge(chs, out);

	// Recover color space
	cv::cvtColor(out, out, CV_Lab2BGR);
	
	// Display result
	cv::namedWindow("Input");
	cv::namedWindow("Output");
	cv::imshow("Input", img);
	cv::imshow("Output", out);
	cv::waitKey(0);
	cv::destroyAllWindows();

	// Save result
	cout << "[FaugerasAlgorithm] save as: ";
	cin >> ofname;
	out.convertTo(out, CV_8UC3, 255.0);
	cv::imwrite(ofname, out);
}