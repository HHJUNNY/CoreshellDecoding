#pragma once
#include <opencv\cv.h>
#include <opencv\highgui.h>

class CWellDetector
{
private:
	IplImage* _pOriginalImage;
	cv::Mat _img;				// obtained from _pOriginalImage
	cv::Mat _imgGray;			// convert gray from _img
	cv::Mat _imgBinary;			// convert binary from _imgGray
	int _nWellNumForPicture;
	double _nWellDiameter;
//	cv::RNG _rng;				// this if for visualization

public:
	CWellDetector(IplImage* picture, int nWellNum, int nWellDiameter);
	~CWellDetector();

	std::vector<cv::Mat> Detect();

private:
	void Convert2Binary();
};