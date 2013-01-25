#include <iostream>
using namespace std;

#include <opencv2\opencv.hpp>

#include "../clcnst/clcnst.h"

int main(int argc, char** argv) {
	if(argc < 3) {
		cout << "usage: HornAlgorithm.exe [input image] [output image] [threshold]" << endl;
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

	float theta = (argc > 3) ? atof(argv[3]) : 0.05f;
	printf("threshold = %f\n", theta);

	cv::Mat out, laplace;

	// 輝度の対数を取る
	clcnst::logarithm(img, out);
	
	// ラプラシアンフィルタをかける
	clcnst::laplacian(out, laplace);

	// 閾値処理
	clcnst::threshold(laplace, laplace, theta);
		
	// ガウス・ザイデル法
	clcnst::gauss_seidel(out, laplace, 20);		

	// 正規化
	clcnst::normalize(out, out);

	// 指数をとる計算
	clcnst::exponential(out, out);
		
	cv::namedWindow("Input");
	cv::namedWindow("Output");
	cv::imshow("Input", img);
	cv::imshow("Output", out);
	cv::waitKey(0);
	cv::destroyAllWindows();

	out.convertTo(out, CV_8UC3, 255.0);
	cv::imwrite(argv[2], out);
}
