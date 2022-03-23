#pragma once
#include <iostream>
#include <fstream>
#include <math.h>
#include <random>
#include <string>
#include <Windows.h> 
#include "Bitmapstuff.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
using namespace std;

static double ar = 16 / 16;// aspect ratio
static const int tamx = 1024;
static int tamy = tamx / ar;
static double rsize = 1024.0 / tamx;
static const int spp = 1;//samples per pixel

typedef struct Tcolor {
	COLORREF* arr;

	Tcolor() {
		arr = new COLORREF[tamx * tamy];
	}

	~Tcolor()
	{
		delete[] arr;
	}
};

__global__ typedef struct Tpair {
	double x;
	double y;
	//Tpair(double f, double s) : x(f), y(s) {};
	__device__ __host__ void fill(double f, double s)
	{
		x = f;
		y = s;
	}
	__device__ void map(const Tpair& norx, const Tpair& nory, const Tpair& maxx, const Tpair& maxy)//Normalizes pair this with max value max to interval nor
	{
		double ot = (x - maxx.x) / (maxx.y - maxx.x);
		x = (((x - maxx.x) * (norx.y - norx.x)) / (maxx.y - maxx.x)) + norx.x;
		y = (((y - maxy.x) * (nory.y - nory.x)) / (maxy.y - maxy.x)) + nory.x;
	}
};

typedef struct Tbounds
{
	double _max;
	double _min;
	double _cen;
	Tbounds(double min = -2.5, double max = 2.5) :_min(min), _max(max), _cen((min + max) / 2) {}

	void scale(double scale)
	{
		_max = _cen + (_max - _cen) * (1 / scale);
		_min = _cen - (_cen - _min) * (1 / scale);
	}

	void move(double per)
	{
		double dist = _max - _min;
		_max += dist * per;
		_min += dist * per;
		_cen = (_min + _max) / 2;
	}

};

//void saveimage(Tcolor* img);

void paint(HDC device, HBITMAP map);

double rand_num();

void saveimage(LPCSTR path, HBITMAP map);

HBITMAP render(Tbounds xbounds, Tbounds ybounds, int it);