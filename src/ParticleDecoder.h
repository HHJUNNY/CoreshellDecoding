#pragma once
#include <string>
#include <vector>

class CParticleDecoder
{
private:
	std::string _sRunPath;
	std::string _sInputFilePrefix;
	std::vector<std::string> _codes;
	int _nWellNumForPicture;
	int _nWellDiameter;
	int _nOption;

public:
	CParticleDecoder(std::string path, std::string option);

	~CParticleDecoder() {}

	void RunProcess();
};

