#include <iostream>
using namespace std;

#include <opencv2\opencv.hpp>

float eps = 0.00001f;

const int offset[4][2] = { {1, 0}, {-1, 0}, {0, 1}, {0, -1} };

int main(int argc, char** argv) {
	if(argc < 3) {
		cout << "usage: BlakeAlgorithm.exe [input image] [output image] [threshold]" << endl;
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

	float theta = argc > 3 ? atof(argv[3]) : 0.003f;
	printf("threshold = %f\n", theta);

	cv::Mat gray, logim, laplace;
	cv::Mat out = cv::Mat(height, width, CV_32FC3);
	for(int c=0; c<channel; c++) {
		// 1チャネル分の情報をコピー
		gray = cv::Mat(height, width, CV_32FC1);
		for(int y=0; y<height; y++) {
			for(int x=0; x<width; x++) {
				gray.at<float>(y, x) = img.at<float>(y, x*channel+c);
			}
		}

		// 輝度の対数をとる
		logim = cv::Mat(height, width, CV_32FC1);
		for(int y=0; y<height; y++) {
			for(int x=0; x<width; x++) {
				logim.at<float>(y, x) = logf(gray.at<float>(y, x) + eps);
			}
		}

		// 勾配場の計算と閾値処理
		cv::Mat grad = cv::Mat::zeros(height, width, CV_32FC2);
		for(int y=0; y<height-1; y++) {
			for(int x=0; x<width-1; x++) {
				float dx = logim.at<float>(y, x+1) - logim.at<float>(y, x);
				float dy = logim.at<float>(y+1, x) - logim.at<float>(y, x);
				if(dx * dx + dy * dy > theta * theta) {
					grad.at<float>(y, x*2+0) = dx;
					grad.at<float>(y, x*2+1) = dy;
				}
			}
		}

		// 勾配場のgradientを計算
		cv::Mat laplace = cv::Mat::zeros(height, width, CV_32FC1);
		for(int y=1; y<height; y++) {
			for(int x=1; x<width; x++) {
				float ddx = grad.at<float>(y, x*2+0) - grad.at<float>(y, (x-1)*2+0);
				float ddy = grad.at<float>(y, x*2+1) - grad.at<float>(y-1, x*2+1);
				laplace.at<float>(y, x) = ddx + ddy;
			}
		}

		// 積分の計算
		cv::Mat T1 = cv::Mat(height, width, CV_32FC1);
		for(int y=0; y<height; y++) {
			for(int x=0; x<width; x++) {
				T1.at<float>(y, x) = logf(img.at<float>(y, x*channel+c) + eps);
			}
		}

		// ガウス・ザイデル法
		cv::Mat T2 = cv::Mat(height, width, CV_32FC1);
		int iter = 20;
		while(iter--) {
			double error = 0.0;
			double div = 0.0;
			for(int y=0; y<height; y++) {
				for(int x=0; x<width; x++) {
					int count = 0;
					float sum = 0.0f;
					for(int i=0; i<4; i++) {
						int xx = x + offset[i][0];
						int yy = y + offset[i][1];
						if(xx >= 0 && yy >= 0 && xx < width && yy < height) {
							sum += T1.at<float>(yy, xx);
							count += 1;
						}
					}
					T2.at<float>(y, x) = (sum - laplace.at<float>(y, x)) / (float)count;
				}
			}
			T2.convertTo(T1, CV_32FC1);
		}

		// 正規化
		float maxval = -100.0f;
		for(int y=0; y<height; y++) {
			for(int x=0; x<width; x++) {
				if(maxval < T2.at<float>(y, x)) {
					maxval = T2.at<float>(y, x);
				}
			}
		}

		printf("maxval = %f\n", maxval);
		for(int y=0; y<height; y++) {
			for(int x=0; x<width; x++) {
				T2.at<float>(y, x) -= maxval;
			}
		}

		// 指数をとる計算		
		for(int y=0; y<height; y++) {
			for(int x=0; x<width; x++) {
				float value = T2.at<float>(y, x);
				out.at<float>(y, x*channel+c) = expf(value - eps);
			}
		}
	}

	cv::namedWindow("Input");
	cv::namedWindow("Output");
	cv::imshow("Input", img);
	cv::imshow("Output", out);
	cv::waitKey(0);
	cv::destroyAllWindows();

	out.convertTo(out, CV_8UC3, 255.0);
	cv::imwrite(argv[2], out);

}