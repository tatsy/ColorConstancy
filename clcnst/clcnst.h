#ifndef _CLCNST_H_
#define _CLCNST_H_

#include <opencv2/opencv.hpp>
#include <cassert>

#if defined(WIN32)		// MS Windows
#define IDAAPI __stdcall
#ifdef __DLL_EXPORT
#define __PORT __declspec(dllexport)
#else
#define __PORT __declspec(dllimport)
#endif
#else
#define __PORT
#endif

// Utility functions for color constancy projects
class clcnst {
private:
	static const int offset[4][2];
	static const float eps;

public:
	// Compute exp of cv::Mat.
	// Input and output can be the same instance.
	// Type of arguments must be CV_32F.
	__PORT static void exponential(cv::Mat& input, cv::Mat& output);

	// Compute log of cv::Mat.
	// Input and output can be the same instance.
	// Type of arguments must be CV_32F.
	__PORT static void logarithm(cv::Mat& input, cv::Mat& output);

	// Solve poisson equation using Gauss-Seldel method.
	__PORT static void gauss_seidel(cv::Mat& I, cv::Mat& L, int maxiter);

	// Apply Laplacian filter.
	// Input and output can be the same instance.
	// Type of arguments must be CV_32FC.
	__PORT static void laplacian(cv::Mat& input, cv::Mat& output);

	// Apply Gaussian filter.
	__PORT static void gaussian(cv::Mat& input, cv::Mat& output, float sigma, int ksize);

	// Apply thresholding operation.
	__PORT static void threshold(cv::Mat& input, cv::Mat& output, float threshold);

	// Normalize output range as the maximum value come to be 1.
	__PORT static void normalize(cv::Mat& input, cv::Mat& output);
	
	// Normalize output range into [lower, upper]
	__PORT static void normalize(cv::Mat& input, cv::Mat& output, float lower, float upper);

	// High emphasis filter
	__PORT static void hef(cv::Mat& input, cv::Mat& output, float lower, float upper, float threshold);
};

#endif
