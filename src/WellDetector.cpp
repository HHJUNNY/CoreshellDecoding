#include "WellDetector.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define M_PI 3.14159265358979323846
#define MAX_BINARY_VALUE 255
#define CONTOUR_AREA_THRESHOLD 25
#define THRESHOLD_TYPE 0

CWellDetector::CWellDetector(IplImage* picture, int nWellNum, int nWellDiameter)
	: _pOriginalImage(picture), _nWellNumForPicture(nWellNum), _nWellDiameter(nWellDiameter)
	//, _rng()			// this if for visualization
{
	_img = cv::cvarrToMat(_pOriginalImage);
}

CWellDetector::~CWellDetector()
{
}

// this if for visualization
void ShowImageNewWindow(const std::string &name, cv::InputArray mat)
{
	cv::namedWindow(name, CV_WINDOW_AUTOSIZE);
	imshow(name, mat);
}

void CWellDetector::Convert2Binary()
{
#pragma region Ori->Gray
	cvtColor(_img, _imgGray, cv::COLOR_BGR2GRAY);
#pragma endregion

#pragma region Gray->Binary
	// make histogram
	int height = _img.rows;
	int width = _img.cols;

	long hist_arr[256] = { 0 };
	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			hist_arr[_imgGray.at<uchar>(i, j)]++;
		}
	}

	// histogram cdf
	long hist_cdf[256] = { 0 };
	hist_cdf[0] = hist_arr[0];
	for (int i = 1; i < 256; ++i)
		hist_cdf[i] = hist_arr[i] + hist_cdf[i - 1];

	int thresholdValue = 0;
	double num_px = height*width;
	double num_px_well = M_PI *_nWellDiameter*_nWellDiameter / 4;
	double ratio = num_px_well*_nWellNumForPicture / num_px;

	// determine threshold value
	for (int i = 255; i >= 0; --i)
	{
		if ((double)hist_cdf[i] / num_px < 1 - ratio)
		{
			thresholdValue = i;
			break;
		}
	}

	threshold(_imgGray, _imgBinary, thresholdValue, MAX_BINARY_VALUE, 0);
	
#pragma endregion
}

std::vector<cv::Mat> CWellDetector::Detect()
{
	std::vector<cv::Mat> returnVector;

	Convert2Binary();
	
#pragma region Find Bounding Rectangles
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	std::vector<int> small_blobs;
	double contour_area;
	cv::Mat temp_image;

	// find all contours in the binary image
	_imgBinary.copyTo(temp_image);
	findContours(temp_image, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	// exclude all small contours
	for (auto it = contours.begin(); it != contours.end(); )
	{
		if (contourArea(*it) < CONTOUR_AREA_THRESHOLD)
			it = contours.erase(it);
		else
			++it;
	}

	std::vector<std::vector<cv::Point>> contours_poly(contours.size());
	std::vector<cv::Rect> boundRect(contours.size());
	std::vector<cv::Rect> topNRect;
	for (size_t i = 0; i < contours.size(); ++i)
	{
		cv::approxPolyDP(cv::Mat(contours[i]), contours_poly[i], 3, true);
		boundRect[i] = cv::boundingRect(cv::Mat(contours_poly[i]));

		if (topNRect.size() < _nWellNumForPicture)
		{
			topNRect.push_back(boundRect[i]);
			std::sort(topNRect.begin(), topNRect.end(),
				[](const cv::Rect & a, const cv::Rect & b)
			{
				return a.area() < b.area();
			});
		}
		else if (boundRect[i].area() > topNRect.front().area())
		{
			topNRect.front() = boundRect[i];
			std::sort(topNRect.begin(), topNRect.end(),
				[](const cv::Rect & a, const cv::Rect & b)
			{
				return a.area() < b.area();
			});
		}
	}

	// sort by x position
	std::sort(topNRect.begin(), topNRect.end(),
		[](const cv::Rect & a, const cv::Rect & b)
	{
		return a.x < b.x;
	});

	for (size_t i = 0; i < topNRect.size(); ++i)
	{
		if (topNRect[i].width > _nWellDiameter * 0.5 && topNRect[i].width < _nWellDiameter * 1.5 &&
			topNRect[i].height > _nWellDiameter * 0.5 && topNRect[i].height < _nWellDiameter * 1.5)
		{
			cv::Point2i offset = cv::Point2i(topNRect[i].width / 2 - _nWellDiameter / 2, topNRect[i].height / 2 - _nWellDiameter / 2);
			cv::Rect roi = cv::Rect(topNRect[i].tl() + offset, topNRect[i].br() - offset);
			roi.width = _nWellDiameter;
			roi.height = _nWellDiameter;
			returnVector.emplace_back(cv::Mat(_img, roi));
		}
	}

#pragma endregion

	return returnVector;
}