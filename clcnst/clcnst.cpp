#define __DLL_EXPORT
#include "clcnst.h"

const int clcnst::offset[4][2]  = { {-1, 0}, {1, 0}, {0, -1}, {0, 1} };

const float clcnst::eps = 0.00001f;

__PORT void clcnst::exponential(cv::Mat& input, cv::Mat& output) {	
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
			for(int c=0; c<channel; c++) {
				o_ptr->at<float>(y, x*channel+c) = exp(i_ptr->at<float>(y, x*channel+c) - clcnst::eps);
			}
		}
	}
}

__PORT void clcnst::logarithm(cv::Mat& input, cv::Mat& output) {	
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
			for(int c=0; c<channel; c++) {
				o_ptr->at<float>(y, x*channel+c) = log(i_ptr->at<float>(y, x*channel+c) + clcnst::eps);
			}
		}
	}
}

__PORT void clcnst::gauss_seidel(cv::Mat& I, cv::Mat& L, int maxiter) {
	int width = I.cols;
	int height = I.rows;
	int channel = I.channels();
	assert(width == L.cols && height == L.rows && channel == L.channels());

	while(maxiter--) {
		for(int c=0; c<channel; c++) {
			for(int y=0; y<height; y++) {
				for(int x=0; x<width; x++) {
					int count = 0;
					float sum = 0.0f;
					for(int i=0; i<4; i++) {
						int xx = x + clcnst::offset[i][0];
						int yy = y + clcnst::offset[i][1];
						if(xx >= 0 && yy >= 0 && xx < width && yy < height) {
							sum += I.at<float>(yy, xx*channel+c);
							count += 1;
						}
					}
					I.at<float>(y, x*channel+c) = (sum - L.at<float>(y, x*channel+c)) / (float)count;
				}
			}
		}
	}
}

__PORT void clcnst::laplacian(cv::Mat&input, cv::Mat& output) {
	cv::Mat* i_ptr = &input;
	int width = i_ptr->cols;
	int height = i_ptr->rows;
	int channel = i_ptr->channels();

	cv::Mat* o_ptr = &output;
	if(i_ptr != o_ptr) {
		*o_ptr = cv::Mat(height, width, CV_MAKETYPE(CV_32F, channel));
	}

	for(int c=0; c<channel; c++) {
		for(int y=0; y<height; y++) {
			for(int x=0; x<width; x++) {
				int count = 0;
				float sum = 0.0f;
				for(int i=0; i<4; i++) {
					int xx = x + clcnst::offset[i][0];
					int yy = y + clcnst::offset[i][1];
					if(xx >= 0 && yy >= 0 && xx < width && yy < height) {
						count += 1;
						sum += i_ptr->at<float>(yy, xx*channel+c);
					}
				}
				o_ptr->at<float>(y, x*channel+c) = sum - (float)count * i_ptr->at<float>(y, x*channel+c);
			}
		}
	}
}

__PORT void clcnst::threshold(cv::Mat& input, cv::Mat& output, float threshold) {
	cv::Mat* i_ptr = &input;
	int width = i_ptr->cols;
	int height = i_ptr->rows;
	int channel = i_ptr->channels();

	cv::Mat* o_ptr = &output;
	if(i_ptr != o_ptr) {
		*o_ptr = cv::Mat(height, width, CV_MAKETYPE(CV_32F, channel));
	}

	for(int c=0; c<channel; c++) {
		for(int x=0; x<width; x++) {
			for(int y=0; y<height; y++) {
				if(fabs(i_ptr->at<float>(y, x*channel+c)) < threshold) {
					o_ptr->at<float>(y, x*channel+c) = 0.0f;
				} else {
					o_ptr->at<float>(y, x*channel+c) = i_ptr->at<float>(y, x*channel+c);
				}
			}
		}
	}
}

__PORT void clcnst::normalize(cv::Mat& input, cv::Mat& output) {
	cv::Mat* i_ptr = &input;
	int width = i_ptr->cols;
	int height = i_ptr->rows;
	int channel = i_ptr->channels();

	cv::Mat* o_ptr = &output;
	if(i_ptr != o_ptr) {
		*o_ptr = cv::Mat(height, width, CV_MAKETYPE(CV_32F, channel));
	}

	for(int c=0; c<channel; c++) {
		float maxval = -100.0f;
		for(int y=0; y<height; y++) {
			for(int x=0; x<width; x++) {
				if(maxval < i_ptr->at<float>(y, x*channel+c)) {
					maxval = i_ptr->at<float>(y, x*channel+c);
				}
			}
		}

		for(int y=0; y<height; y++) {
			for(int x=0; x<width; x++) {
				o_ptr->at<float>(y, x*channel+c) = i_ptr->at<float>(y, x*channel+c) - maxval;
			}
		}
	}
}
