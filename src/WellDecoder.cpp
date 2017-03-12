#include "WellDecoder.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/text.hpp>
#include <opencv2/core/utility.hpp>
#include "voronoi\voronoi.h"

using namespace std;
using namespace cv::text;

#define MAX_BINARY_VALUE 255

class VoronoiIterator {
public:
	VoronoiIterator() {}

	void init(const cv::Mat1b & query,
		string implementation_name_) {
		_implementation_name = implementation_name_;
		_first_img = query.clone();
		_bounding_box = VoronoiThinner::copy_bounding_box_plusone(query, _first_img, true);
		_curr_skel = _first_img.clone();
		// printf("first_img:%s\n", image_utils::infosImage(_first_img).c_str());
		_curr_iter = 1;
		_nframes = 0;
	}

	inline cv::Mat1b first_img() {
		return _first_img.clone();
	}

	cv::Mat1b current_skel() const {
		return _thinner.get_skeleton().clone();
	}
	
	inline cv::Mat1b contour_brighter(const cv::Mat1b & img) {
		_contour_viz.from_image_C4(img);
		cv::Mat1b ans;
		_contour_viz.copyTo(ans);
		return ans;
	}

	inline cv::Mat3b contour_color(const cv::Mat1b & img) {
		_contour_viz.from_image_C4(img);
		return _contour_viz.illus().clone();
	}

	//! \return true if success
	bool iter() {
		++_nframes;
		bool reuse = (_implementation_name != IMPL_MORPH); // cant reuse with morph implementation
		bool success = false;
		if (reuse)
			success = _thinner.thin(_curr_skel, _implementation_name, false, 1); // already cropped
		else
			success = _thinner.thin(_first_img, _implementation_name, false, _nframes); // already cropped
		_thinner.get_skeleton().copyTo(_curr_skel);
		return success;
	}

	inline bool has_converged() const { return _thinner.has_converged(); }

	inline int cols() const { return _first_img.cols; }
	inline int rows() const { return _first_img.rows; }
	inline int frames() const { return _nframes; }
	inline cv::Rect bounding_box() const { return _bounding_box; }

	//protected:
	string _implementation_name;
	bool _crop_img_before;
	int _nframes;
	int _curr_iter;
	cv::Mat1b _first_img;
	cv::Mat1b _curr_skel;
	cv::Rect _bounding_box;
	VoronoiThinner _thinner;
	ImageContour _contour_viz;
}; // end class VoronoiIterator

#if DEBUG
#define ROW 3
#define COL 3
#endif

CWellDecoder::CWellDecoder(int wellSize, vector<string> &possibleCodes)
{
	_vocabulary = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"; // must have the same order as the clasifier output classes
	_ocr = loadOCRHMMClassifierCNN("OCRBeamSearch_CNN_model_data.xml.gz");
	_possibleCodes = &possibleCodes;

	for (auto code : *_possibleCodes)
	{
		vector<int> letters;
		for (auto letter : code)
		{
			if (letter == '1')
			{
				// ignore
			}
			else if (letter == 'B')
			{
				letters.push_back(_vocabulary.find(letter));
				// 8 ~ B
				letters.push_back(_vocabulary.find('8'));
			}
			else
			{
				letters.push_back(_vocabulary.find(letter));
			}
		}
		_possibleLetters.push_back(letters);
	}
}

CWellDecoder::~CWellDecoder()
{
}

/** @brief Display input image on window in sequence.

@param src Image to be displayed.
*/
void CWellDecoder::DisplayImage(cv::Mat &src)
{
#if DEBUG
	if (_dispImg.rows == 0)
		_dispImg = cv::Mat(src.rows * ROW, src.cols * COL, CV_8UC3, cv::Scalar(0, 0, 0));

	if (src.rows != _dispImg.rows / ROW || src.cols != _dispImg.cols / COL)
		cv::resize(src, src, cv::Size(_dispImg.rows / ROW, _dispImg.cols / COL));

	cv::Rect destRect = cv::Rect(src.rows*_nRow++, src.cols*_nCol, src.rows, src.cols);
	cv::Mat destMat = _dispImg(destRect);
	if (src.type() == CV_8UC1)
	{
		cv::Mat cvt;
		cv::cvtColor(src, cvt, cv::COLOR_GRAY2BGR);
		cvt.copyTo(destMat);
	}
	else if (src.type() == CV_8UC3)
	{
		src.copyTo(destMat);
	}
	else
	{
		perror("[SYSTEM] Error in displaying image.");
	}

	if (_nRow == ROW)
	{
		_nRow = 0;
		++_nCol;
	}

	cv::namedWindow("Images", CV_WINDOW_AUTOSIZE);
	cv::imshow("Images", _dispImg);
	cv::waitKey(0);
#endif
}

#define THRESHOLDING_BLOCKSIZE_DENOMINATOR 10
#define THRESHOLDING_CONSTANT -0.5
string CWellDecoder::Decode(IplImage* picture, double codeHeight, cv::Mat &outputImage)
{
	_nRow = 0, _nCol = 0;

	cv::Mat src = cv::cvarrToMat(picture);
	DisplayImage(src);

	// convert to grayscale
	cv::Mat dstGray;
	cv::cvtColor(src, dstGray, cv::COLOR_BGR2GRAY);
	//DisplayImage(dstGray);

	// Thresholding
	cv::Mat dstThrs;
	int blockSize = src.rows / THRESHOLDING_BLOCKSIZE_DENOMINATOR;
	if (blockSize % 2 == 0) { --blockSize; }
	adaptiveThreshold(dstGray, dstThrs, MAX_BINARY_VALUE, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, blockSize, THRESHOLDING_CONSTANT);
	DisplayImage(dstThrs);

	// Denoise
	cv::Mat dstClean;
	RemoveSmallBlobs(dstThrs, dstClean);

	// find indicator in advance
	auto indicator = FindIndicator(dstClean, nullptr);

	// Extract skeleton
	cv::Mat dstSkel;
	ExtractSkeleton(dstThrs, dstSkel);
	//DisplayImage(dstSkel);
	
	// find indicator line & rotate image along indicator
	cv::Mat dstRot;
	double indicator_y = RotateHorizontally(dstSkel, dstRot, indicator, codeHeight);
	//DisplayImage(dstRot);

	// Adjust morphology after rotation.
	// Usually, horizontal lines of core-shell graphical codes are thinner than vertical lines of those.
	// Thus, dilating vertically, and then eroding to conserve total area.
	cv::Mat mask = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 5), cv::Point(1, 1));
	cv::dilate(dstRot, dstRot, mask, cv::Point(-1, -1), 1);
	//DisplayImage(dstRot);
	mask = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3), cv::Point(1, 1));
	cv::erode(dstRot, dstRot, mask, cv::Point(-1, -1), 1);
	DisplayImage(dstRot);
	
	// detect character
	vector<cv::Rect> charBoundRect;
	DetectCharacters(dstRot, charBoundRect, codeHeight, indicator_y);

	// decode
	return RunOCR(dstRot, charBoundRect, outputImage);
}

/** @brief Sample distances between the lines which are parallel to indicator.

@param picture Source image.
@param gapsOutput Distances Sampled.
*/
void CWellDecoder::DistanceSampling(IplImage* picture, vector<double>& gapsOutput)
{
	_nRow = 0, _nCol = 0;

	cv::Mat src = cv::cvarrToMat(picture);
	cv::Mat dstClean;

	// convert to grayscale
	cv::Mat dstGray;
	cv::cvtColor(src, dstGray, cv::COLOR_BGR2GRAY);

	// Thresholding
	cv::Mat dstThrs;
	int blockSize = src.rows / THRESHOLDING_BLOCKSIZE_DENOMINATOR;
	if (blockSize % 2 == 0) { --blockSize; }
	adaptiveThreshold(dstGray, dstThrs, MAX_BINARY_VALUE, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, blockSize, THRESHOLDING_CONSTANT);

	// remove small blobs
	RemoveSmallBlobs(dstThrs, dstClean);

	vector<double> sample;
	FindIndicator(dstClean, &sample);
	gapsOutput.insert(gapsOutput.end(), sample.begin(), sample.end());
}

#define DENOISING_MINIMUM_AREA_DENOMINATOR 500
/** @brief Extract topological skeleton features from binary image using thinning algorithm designed by Zhang-Suen

@param src Source image.
@param dst Destination image.
*/
void CWellDecoder::ExtractSkeleton(cv::Mat &src, cv::Mat &dst)
{
	// pre-processing: apply erode & dilate action to erase small noise and reconnect broken letters
	cv::Mat mask = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(1, 1));
	//cv::dilate(src, dst, mask, cv::Point(-1, -1), 2);
	cv::erode(src, dst, mask, cv::Point(-1, -1), 1);
	cv::dilate(dst, dst, mask, cv::Point(-1, -1), 1);

	// Denoise
	cv::Mat dstClean;
	RemoveSmallBlobs(dst, dstClean);

	cv::dilate(dstClean, dst, mask, cv::Point(-1, -1), 1);
	
	// init iterator
	VoronoiIterator it;
	it.init(dst, IMPL_ZHANG_SUEN_ORIGINAL);
	
	// loop
	while (it.frames() < 1) {
		it.iter();
		it.current_skel().copyTo(dst(it.bounding_box())); 
	}
	DisplayImage(dst);

	cv::dilate(dst, dst, mask, cv::Point(-1, -1), 1);

	/*
	it.init(dst, IMPL_GUO_HALL_ORIGINAL);
	while (it.frames() < 1) {
		it.iter();
		it.current_skel().copyTo(dst(it.bounding_box()));
	}*/
}

/** @brief Remove small contours to reduce noise.

@param src Source image.
@param dst Destination image.
*/
void CWellDecoder::RemoveSmallBlobs(cv::Mat &src, cv::Mat &dst)
{
	src.copyTo(dst);

	// step 1. erase small noise (white area)
	vector<vector<cv::Point>> contours;
	vector<cv::Vec4i> hierarchy;
	vector<int> small_blobs;
	double contour_area;
	cv::Mat temp_image;

	// find all contours in the binary image
	src.copyTo(temp_image);
	cv::findContours(temp_image, contours, hierarchy, CV_RETR_CCOMP,
		CV_CHAIN_APPROX_SIMPLE);

	// Find indices of contours whose area is less than `threshold` 
	if (!contours.empty()) {
		for (size_t i = 0; i<contours.size(); ++i) {
			contour_area = cv::contourArea(contours[i]);
			// threashold specifying minimum area of a blob
			if (contour_area < src.rows * src.cols / DENOISING_MINIMUM_AREA_DENOMINATOR)
				small_blobs.push_back(i);
		}
	}

	// fill-in all small contours with zeros
	for (size_t i = 0; i < small_blobs.size(); ++i) {
		cv::drawContours(dst, contours, small_blobs[i], cv::Scalar(0),
			CV_FILLED, 8);
	}
}

#define PARALELL_ANGLE_TOLERANCE 5
#define INDICATOR_DISTANCE_MINIMUM_DENOMINATOR 20
/** @brief Find an indicator, the longest straitfoward line in particle.

@param src Source image.
@param gaps Distances between the indicator and lines which are parallel to the indicatger.
*/
cv::Vec4i CWellDecoder::FindIndicator(cv::Mat &src, vector<double> *gaps)
{
	// find indicator and get tilted angle in small region first
	//DisplayImage(src);
	cv::Mat indicator_roi = src(cv::Rect(src.rows / 5, src.cols / 5, src.rows * 0.6, src.cols * 0.6));
	//DisplayImage(indicator_roi);

	vector<cv::Vec4i> lines;
	HoughLinesP(indicator_roi, lines, 1, CV_PI / 180, 80, src.rows / 4, 5);

	cv::Vec4i max_l;
	double max_dist = -1.0;
	for (size_t i = 0; i < lines.size(); i++)
	{
		cv::Vec4i l = lines[i];
		double theta1, theta2, hyp;

		theta1 = (l[3] - l[1]);
		theta2 = (l[2] - l[0]);
		hyp = hypot(theta1, theta2);

		if (max_dist < hyp) {
			max_l = l;
			max_dist = hyp;
		}
	}
	double indi_angle = atan2(max_l[3] - max_l[1], max_l[2] - max_l[0]) * 180.0 / CV_PI;
	
	// find longest line (=Indicator) in entire well image
	HoughLinesP(src, lines, 1, CV_PI / 180, 80, src.rows / 3, 10);
	//cv::Mat cvt;
	//cv::cvtColor(src, cvt, cv::COLOR_GRAY2BGR);
	max_l = 0;
	max_dist = -1.0;
	for (size_t i = 0; i < lines.size(); i++)
	{
		cv::Vec4i l = lines[i];
		double theta1, theta2, hyp;

		theta1 = (l[3] - l[1]);
		theta2 = (l[2] - l[0]);
		hyp = hypot(theta1, theta2);
		double angle = atan2(theta1, theta2) * 180.0 / CV_PI;

		if (abs(indi_angle - angle) <= PARALELL_ANGLE_TOLERANCE || abs(abs(indi_angle - angle) - 180.0) <= PARALELL_ANGLE_TOLERANCE)
		{
			if (max_dist < hyp) {
				max_l = l;
				max_dist = hyp;
			}

			//line(cvt, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 0, MAX_BINARY_VALUE), 1);
		}

	}
	//DisplayImage(cvt);

	if (nullptr != gaps)
	{
		double indi_angle = atan2(max_l[3] - max_l[1], max_l[2] - max_l[0]) * 180.0 / CV_PI;

		// calculate and collect gaps between the indicator and other lines which are roughly parallel to the indicator
		cv::Point2d max_l_begin(max_l[0], max_l[1]);
		cv::Point2d max_l_end(max_l[2], max_l[3]);
		cv::Point2d v1 = max_l_end - max_l_begin;
		for (size_t i = 0; i < lines.size(); i++)
		{
			cv::Vec4i l = lines[i];
			if (l == max_l)
				continue;

			double theta1, theta2, hyp;

			theta1 = (l[3] - l[1]);
			theta2 = (l[2] - l[0]);
			double angle = atan2(theta1, theta2) * 180.0 / CV_PI;
			hyp = hypot(theta1, theta2);

			// if two lines are parallel within tolerance range
			if (hyp >= max_dist / 2 &&
				(abs(indi_angle - angle) <= PARALELL_ANGLE_TOLERANCE || abs(abs(indi_angle - angle) - 180.0) <= PARALELL_ANGLE_TOLERANCE))
			{
				cv::Point2d center_l((l[0] + l[2]) / 2, (l[1] + l[3]) / 2);
				// calc distance
				cv::Point2d v2 = center_l - max_l_begin;
				double distance = abs(v1.cross(v2)) / max_dist;

				// lower limit: well size / INDICATOR_DISTANCE_MINIMUM_DENOMINATOR (maybe it should be adjusted..)
				if (distance > (double)src.rows / INDICATOR_DISTANCE_MINIMUM_DENOMINATOR)
				{
					gaps->push_back(distance);
				}
			}
		}
	}

	return max_l;
}

/** @brief Rotate the image in order to align the indicator horizontally

@param src Source image.
@param dst Destination image.
@param srcRaw Original(color) source image.
@param dstRaw Destination image to rotate origianl source image.
@param indicator Indicator.
@param codeHeight Vertical distance between indicators.

@returns y-coordinate of the main indicator after the rotation.
*/
double CWellDecoder::RotateHorizontally(cv::Mat &src, cv::Mat &dst, cv::Vec4i indicator, double codeHeight)
{
	// rotate Image along indicator
	double indi_angle = atan2(indicator[3] - indicator[1], indicator[2] - indicator[0]) * 180.0 / CV_PI;
	double indi_angle_rad = indi_angle / 180.0*CV_PI;

	cv::Point2f center(src.cols / 2., src.rows / 2.);
	cv::Mat rotMat1 = getRotationMatrix2D(center, indi_angle, 1.0);
	warpAffine(src, dst, rotMat1, src.size());

	// erase indicator and adjacent regions
	double indi_rotated_y = (indicator[0] - center.x) * sin(-indi_angle_rad) + (indicator[1] - center.y) * cos(-indi_angle_rad) + center.y;
	return indi_rotated_y;
}


#define LOWER_HEIGHT_RATIO 0.400
#define UPPER_HEIGHT_RATIO 0.833
#define LOWER_WIDTH_RATIO 0.100
#define UPPER_WIDTH_RATIO 0.700
#define INDICATOR_REGION_THICKNESS_RATIO 0.125
#define MAXIMUM_DISTANCE_FROM_CENTER_RATIO 1.5
/** @brief Detect characters and their bounding rectangles in a particle.

@param src Source image. The indicator in the image mush be horizontal.
@param charBoundRect Output rectangles information that bounding detected characters.
@param codeHeight Vertical distance between indicators.
@param indicator_y y-coordinate of the indicator.
*/
void CWellDecoder::DetectCharacters(cv::Mat &src, vector<cv::Rect> &charBoundRect, int codeHeight, int indicator_y)
{
	cv::Point2d wellCenter(src.rows / 2, src.cols / 2);
	vector<vector<cv::Point>> contours; 
	vector<cv::Vec4i> hierarchy;

	/// Find contours
	findContours(src, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	/// Approximate contours to polygons + get bounding rects
	vector<vector<cv::Point>> contours_poly(contours.size());

	// Filtering appropriate contours only
	for (int i = 0; i < contours.size(); i++)
	{
		cv::approxPolyDP(cv::Mat(contours[i]), contours_poly[i], 3, true);
		auto boundRect = cv::boundingRect(cv::Mat(contours_poly[i]));

		// Criterion 1. size
		if (boundRect.width > codeHeight * UPPER_WIDTH_RATIO ||
			boundRect.width < codeHeight * LOWER_WIDTH_RATIO ||
			boundRect.height > codeHeight * UPPER_HEIGHT_RATIO ||
			boundRect.height < codeHeight * LOWER_HEIGHT_RATIO)
		{
			continue;
		}

		// Criterion 2. Locate between two adjacent indicators
		int center_y = boundRect.y + boundRect.height / 2;
		int center_to_indicator1 = abs((center_y - indicator_y) % codeHeight);
		int center_to_indicator2 = codeHeight - center_to_indicator1;
		if (center_to_indicator1 < boundRect.height / 2 + codeHeight * INDICATOR_REGION_THICKNESS_RATIO ||
			center_to_indicator2 < boundRect.height / 2 + codeHeight * INDICATOR_REGION_THICKNESS_RATIO)
		{
			continue;
		}

		// Criterion 3. Locate near center
		cv::Point2d codeCenter((boundRect.br() + boundRect.tl()) / 2);
		if (cv::norm(wellCenter - codeCenter) > codeHeight * MAXIMUM_DISTANCE_FROM_CENTER_RATIO)
		{
			continue;
		}

		// TODO: Criterion 4. If aspect ratio (width/height) of certain code is too small,
		// then it has to have enough enough blank space beside it
				

		charBoundRect.push_back(boundRect);
	}

	// Sort contours aslike 2d array
	auto sortRuleLambda = [codeHeight, indicator_y](const cv::Rect & a, const cv::Rect & b) -> bool
	{
		int val_a_y = (a.y - indicator_y % codeHeight) / codeHeight;
		int val_b_y = (b.y - indicator_y % codeHeight) / codeHeight;
		return val_a_y == val_b_y ? a.x < b.x : val_a_y < val_b_y;
	};
	std::sort(charBoundRect.begin(), charBoundRect.end(), sortRuleLambda);

	// if two adjacent contours are close enough to each other, or overlapped, than combine them
	int i = 0;
	if (charBoundRect.size() >= 2)
	{
		auto left = charBoundRect.begin();
		do
		{
			auto right = left;
			++right;

			if (right == charBoundRect.end())
				break;

			// if a gap exists between the two Rects than ignore
			if (left->br().x < right->x && (right->x - left->br().x) > 5)
			{
				++left;
				continue;
			}

			// combine the two Rects
			cv::Rect combinedRect(cv::Point(min(left->x, right->x), min(left->y, right->y)), cv::Point(max(left->br().x, right->br().x), max(left->br().y, right->br().y)));

			// if combined box is too large then ignore
			if (combinedRect.width > codeHeight * UPPER_WIDTH_RATIO ||
				combinedRect.width < codeHeight * LOWER_WIDTH_RATIO ||
				combinedRect.height > codeHeight * UPPER_HEIGHT_RATIO ||
				combinedRect.height < codeHeight * LOWER_HEIGHT_RATIO)
			{
				++left;
				continue;
			}

			// combine
			*left = combinedRect;
			charBoundRect.erase(right);
		} while (true);
	}

	/// Draw polygonal contour + bonding rects + circles
	cv::Mat drawing;
 	cv::cvtColor(src, drawing, cv::COLOR_GRAY2BGR);
	for (int i = 0; i< charBoundRect.size(); i++)
	{
		rectangle(drawing, charBoundRect[i].tl(), charBoundRect[i].br(), cv::Scalar(0, 0, 255), 2, 8, 0);
	}

	DisplayImage(drawing);
}

vector<double> conf_bound = { 0.90, 0.70, 0.50, 0.25, 0.00 };

/** @brief Identify code encoded in particle via ocr process.

@param src Source image. The indicator in the image mush be horizontal.
@param srcRaw Original(color) source image. The indicator in the image mush be horizontal.
@param dst Destination image with drawing box containing code.
@param codeHeight Vertical distance between indicators.
@param outputImage Destination image with drawing box containing code.

@returns Identified code. It returns string.empty() if the identification is failed.
*/
string CWellDecoder::RunOCR(cv::Mat &src, vector<cv::Rect> &charBoundRect, cv::Mat &outputImage)
{
	///						 0							26								52				62
	/// 	_vocabulary =	"abcdefghijklmnopqrstuvwxyz	A	BCDEFGHIJKLMNOPQRSTUVWXYZ	0	12345678	9";
	cv::cvtColor(src, outputImage, cv::COLOR_GRAY2BGR);

	cv::Mat ROI;
	cv::Mat croppedImage;

	vector<int> out_classes;
	vector<double> out_confidences;
	double max_confidence = -1;
	int max_rect_index = 0;
	int max_conf_bound_index = 0;
	int max_code_index;

	vector<vector<int>> classificationCount_1;
	vector<vector<int>> classificationCount_2;

	vector<vector<int>>* selected_c_count;
	int up_or_down = 0;

	for (auto itr : *_possibleCodes)
	{
		vector<int> counts;
		for (size_t i = 0; i < conf_bound.size(); ++i)
		{
			counts.push_back(0);
		}
		classificationCount_1.push_back(counts);
		classificationCount_2.push_back(counts);
	}

	for (size_t r = 0; r < charBoundRect.size(); ++r)
	{
		// original
		cv::Mat ROI(src, charBoundRect[r]);
		ROI.copyTo(croppedImage);
		selected_c_count = &classificationCount_1;
		for (int up_down = 0; up_down < 2; ++up_down)
		{
			_ocr->eval(croppedImage, out_classes, out_confidences);
			
			for (size_t t = 0; t < _possibleLetters.size(); ++t)
			{
				for (auto letter : _possibleLetters[t])
				{
					auto index = find(out_classes.begin(), out_classes.end(), letter) - out_classes.begin();
					for (size_t i = 0; i < conf_bound.size(); ++i)
					{
						if (out_confidences[index] > conf_bound[i])
						{
							++(*selected_c_count)[t][i];

							if (out_confidences[index] > max_confidence)
							{
								max_rect_index = r;
								max_confidence = out_confidences[index];
								max_conf_bound_index = i;
								max_code_index = t;
								up_or_down = up_down;
							}
							break;
						}
					}
				}
			}

			// upside down
			selected_c_count = &classificationCount_2;
			cv::flip(croppedImage, croppedImage, -1);
		}
	}

	selected_c_count = up_or_down == 0 ? &classificationCount_1 : &classificationCount_2;

	// Check if one code monopolies result of highest confidence group.
	if (max_confidence > 0 && max_conf_bound_index < conf_bound.size() - 2)
	{
		int highest_conf_count_others = -(*selected_c_count)[max_code_index][max_conf_bound_index];
		int next_highest_conf_count_others = -(*selected_c_count)[max_code_index][max_conf_bound_index+1];
		for (size_t i = 0; i < selected_c_count->size(); ++i)
		{
			highest_conf_count_others += (*selected_c_count)[i][max_conf_bound_index];
			next_highest_conf_count_others += (*selected_c_count)[i][max_conf_bound_index + 1];
		}

		if (highest_conf_count_others == 0 && next_highest_conf_count_others == 0)
		{
			rectangle(outputImage, charBoundRect[max_rect_index].tl(), charBoundRect[max_rect_index].br(), cv::Scalar(0, 0, 255), 2, 8, 0);

			if (up_or_down == 1)
			{
				cv::flip(outputImage, outputImage, -1);
			}

			return (*_possibleCodes)[max_code_index];
		}
	}

	return "Unidentified";
}