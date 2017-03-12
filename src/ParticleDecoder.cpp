#include "ParticleDecoder.h"
#include "WellDetector.h"
#include "WellDecoder.h"
#include <direct.h>
#include <io.h>
#include <fstream>
#include <sstream>
#include <opencv\cv.h>
#include <opencv2/opencv.hpp>
#include <opencv\highgui.h>
#include <thread>

#define SAMPLING_SIZE 200

const std::string WELL_IMAGE_DIR = "DetachedWell\\";

CParticleDecoder::CParticleDecoder(std::string path, std::string option)
	: _sRunPath(path), _nWellNumForPicture(0), _nOption(0)
{
	std::string line;
	std::ifstream input(path + "info.txt");

	if (input.fail())
	{
		std::cout << "Cannot find info.txt" << std::endl;
		system("pause");
	}
	else
	{
		for (int i = 0; i < option.length(); ++i)
		{
			switch (option[i])
			{
			case 'T':
			case 't':
				_nOption = _nOption | 0x01;
				break;
			case 'C':
			case 'c':
				_nOption = _nOption | 0x10;
				break;
			default:
				break;
			}
		}

		std::vector<std::string> argu;
		while (std::getline(input, line))
		{
			argu.push_back(line);
		}

		for (size_t i = 0; i < argu.size(); i += 2)
		{
			if (argu[i].compare("input prefix") == 0)
			{
				_sInputFilePrefix = argu[i+1];
			}
			else if (argu[i].compare("well count in a picture") == 0)
			{
				_nWellNumForPicture = atoi(argu[i + 1].c_str());
			}
			else if (argu[i].compare("well diameter pixel") == 0)
			{
				_nWellDiameter = atoi(argu[i + 1].c_str());
			}
			else if (argu[i].compare("code list") == 0)
			{
				char delim = ' ';
				auto j = 0;
				auto pos = argu[i + 1].find(delim);
				while (pos != std::string::npos) {
					_codes.push_back(argu[i + 1].substr(j, pos - j));
					j = ++pos;
					pos = argu[i + 1].find(delim, pos);

					if (pos == std::string::npos)
						_codes.push_back(argu[i + 1].substr(j, argu[i + 1].length()));
				}
			}
		}
	}
}

void WellDetectTask(std::string sRunPath, int nWellNumForPicture, int nWellDiameter, std::string sInputFilePrefix,
	 std::string fileName, std::size_t findResult)
{
	IplImage * image = cvLoadImage((sRunPath + fileName).c_str());
	CWellDetector wellDetector(image, nWellNumForPicture, nWellDiameter);
	auto wells = wellDetector.Detect();
	std::string substring = fileName.erase(findResult, sInputFilePrefix.length());
	substring = substring.substr(0, substring.find("."));
	for (std::size_t i = 0; i < wells.size(); ++i)
	{
		std::string newFileName(sRunPath + WELL_IMAGE_DIR + substring + "_" + std::to_string(i) + ".jpg");
		cv::imwrite(newFileName, wells[i]);
	}
}

/// process start from here
void CParticleDecoder::RunProcess()
{
	// Step 1. Detach every single well
	_finddata_t fileinfo;
	struct stat st = { 0 };
	// Decode only option check
	if (!(_nOption & 0x10))
	{
		// Skip when the output directory already exists.
		if (stat((_sRunPath + WELL_IMAGE_DIR).c_str(), &st) == -1)
		{
			_mkdir((_sRunPath + WELL_IMAGE_DIR).c_str());

			intptr_t handle = _findfirst((_sRunPath + "*.tif").c_str(), &fileinfo);
			if (handle == -1)  // No files or directories found
			{
				perror("[SYSTEM] Error searching for file");
				system("pause");
			}

			std::vector<std::thread> threads;
			do
			{
				std::string fileName(fileinfo.name);
				auto findResult = fileName.find(_sInputFilePrefix);
				if (findResult != std::string::npos)
				{
					threads.emplace_back(std::thread(WellDetectTask, _sRunPath, _nWellNumForPicture, _nWellDiameter, _sInputFilePrefix, fileName, findResult));
				}
			} while (_findnext(handle, &fileinfo) == 0);

			_findclose(handle); // Close the stream

			for (int i = 0; i < threads.size(); ++i)
				threads[i].join();
		}
	}

	// Step 2. Decode
	// Detach only option check
	if (!(_nOption & 0x01))
	{
		intptr_t handle = _findfirst((_sRunPath + WELL_IMAGE_DIR + "*.jpg").c_str(), &fileinfo);

		// extract indicator gap height
		CWellDecoder wellDecoder(_nWellDiameter, _codes);
		std::vector<double> gaps;
		do
		{
			std::string fileName(fileinfo.name);

			IplImage * image = cvLoadImage((_sRunPath + WELL_IMAGE_DIR + fileName).c_str());
			wellDecoder.DistanceSampling(image, gaps);

		} while (_findnext(handle, &fileinfo) == 0 && gaps.size() < SAMPLING_SIZE);

		// simple 1d 1peak gaussian distribution estimation
		std::sort(gaps.begin(), gaps.end());
		double mean_distance = 0;
		int begin = SAMPLING_SIZE / 5, end = SAMPLING_SIZE * 4 / 5;

		for (size_t i = begin; i < end; ++i)
		{
			mean_distance += gaps[i];
		}
		mean_distance = mean_distance / (end - begin);

		_findclose(handle); // Close the stream

		// code detecting
		handle = _findfirst((_sRunPath + WELL_IMAGE_DIR + "*.jpg").c_str(), &fileinfo);
		do
		{
			std::string fileName(fileinfo.name);
			IplImage * image = cvLoadImage((_sRunPath + WELL_IMAGE_DIR + fileName).c_str());
			cv::Mat imageProcessed;
			auto result = wellDecoder.Decode(image, mean_distance, imageProcessed);

			if (stat((_sRunPath + result).c_str(), &st) == -1)
			{
				_mkdir((_sRunPath + result).c_str());
			}

			std::string newFileName(_sRunPath + result + "\\" + fileName + ".jpg");
			cv::imwrite(newFileName, imageProcessed);

		} while (_findnext(handle, &fileinfo) == 0);

		_findclose(handle); // Close the stream
	}
}

int main(int argc, char *argv[])
{
	std::string sTargetDirectory;
	std::string sOption;

	if (argc == 1)
	{
		std::cout << "\tPlease input target forder and options." << std::endl << std::endl;
		std::cout << "\t[Option list]" << std::endl;
		std::cout << "\t/t\t\tConduct de\"t\"aching process only." << std::endl;
		std::cout << "\t/c\t\tConduct de\"c\"oding process only." << std::endl << std::endl;;
		std::string line;
		std::string argument;
		std::getline(std::cin, line);
		std::istringstream iss(line);
		iss >> sTargetDirectory;
		while (iss >> argument)
		{
			if (argument.length() == 2 && argument[0] == '/')
			{
				sOption += argument[1];
			}
		}
	}
	else
	{
		sTargetDirectory = argv[1];
		for (int i = 2; i < argc; ++i)
		{
			if (sizeof(argv[i]) == 2 && argv[i][0] == '/')
			{
				sOption += argv[i][1];
			}
		}
	}

	char cCurrentPath[FILENAME_MAX];
	_getcwd(cCurrentPath, sizeof(cCurrentPath));
	std::string fullPath = std::string(cCurrentPath) + "\\" + sTargetDirectory + "\\";

	// check existence
	struct stat st = { 0 };
	while (stat(fullPath.c_str(), &st) == -1)
	{
		std::cout << "\tThere is no \"" << sTargetDirectory << "\" directory. Please try again." << std::endl << std::endl;

		sOption.clear(); sTargetDirectory.clear();
		std::string line;
		std::string argument;
		std::getline(std::cin, line);
		std::istringstream iss(line);
		iss >> sTargetDirectory;
		while (iss >> argument)
		{
			if (argument.length() == 2 && argument[0] == '/')
			{
				sOption += argument[1];
			}
		}
		fullPath = std::string(cCurrentPath) + "\\" + sTargetDirectory + "\\";
	} 
	
	// Create ParticleDecoder object with target path name. 
	CParticleDecoder decoder(fullPath, sOption);
	decoder.RunProcess();
	
	perror("\tAll process finished");
	system("pause");
}