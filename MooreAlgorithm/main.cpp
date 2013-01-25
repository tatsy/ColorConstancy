#include <iostream>
using namespace std;

#include <opencv2\opencv.hpp>

float eps = 0.00001f;

const int offset[4][2] = { {1, 0}, {-1, 0}, {0, 1}, {0, -1} };

int main(int argc, char** argv) {
	if(argc < 3) {
		cout << "usage: MooreAlgorithm.exe [input image] [output image] [threshold]" << endl;
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

}