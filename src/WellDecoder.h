#pragma once
#include <opencv\cv.h>
#include <opencv2/text/ocr.hpp>
#include <string>

class CWellDecoder
{
private:
	cv::Mat _dispImg;
	int _nRow = 0, _nCol = 0;
	std::string _vocabulary;
	std::vector<std::string> *_possibleCodes;
	std::vector<std::vector<int>> _possibleLetters;
	cv::Ptr<cv::text::OCRHMMDecoder::ClassifierCallback> _ocr;

public:
	CWellDecoder(int wellSizet, std::vector<std::string> &possibleCodes);
	~CWellDecoder();

	std::string Decode(IplImage* picture, double codeHeight, cv::Mat &outputImage);
	void DistanceSampling(IplImage* picture, std::vector<double>& gapsOutput);

private:
	void ExtractSkeleton(cv::Mat &src, cv::Mat &dst);
	void RemoveSmallBlobs(cv::Mat &src, cv::Mat &dst);
	cv::Vec4i FindIndicator(cv::Mat &src, std::vector<double> *gaps);
	double RotateHorizontally(cv::Mat &src, cv::Mat &dst, cv::Vec4i indicator, double codeHeight);
	void DetectCharacters(cv::Mat &src, std::vector<cv::Rect> &charBoundRect, int codeHeight, int indicator_y);

	std::string RunOCR(cv::Mat &src, std::vector<cv::Rect> &charBoundRect, cv::Mat &outputImage);

	void DisplayImage(cv::Mat &src);
};

