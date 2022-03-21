#pragma once
#include <iostream>
#include <fstream>
#include <math.h>
#include <random>
#include <string>
#include <Windows.h> 
#include "Bitmapstuff.h"
using namespace std;

double ar = 16 / 16;// aspect ratio
int tamx = 1000;
int tamy = tamx / ar;
double rsize = 1000.0 / tamx;
int it = 10;//iterations to calculate
const int spp = 1;//samples per pixel

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

typedef struct Tpair {
	double x;
	double y;
	Tpair(double f, double s) : x(f), y(s) {};
	Tpair map(Tpair norx, Tpair nory, Tpair maxx, Tpair maxy)//Normalizes pair this with max value max to interval nor
	{
		double ot = (x - maxx.x) / (maxx.y - maxx.x);
		return Tpair((((x - maxx.x) * (norx.y - norx.x)) / (maxx.y - maxx.x)) + norx.x, (((y - maxy.x) * (nory.y - nory.x)) / (maxy.y - maxy.x)) + nory.x);
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
		_max = _cen + (_max - _cen) * (1/scale);
		_min = _cen - (_cen - _min) * (1/scale);
	}

	void move(double per)
	{
		double dist = _max - _min;
		_max += dist * per;
		_min += dist * per;
		_cen = (_min + _max) / 2;
	}

};


Tbounds xbounds;
Tbounds ybounds;
dgdfg  asdakdjsfhksdjf N3
int converges(Tpair coor);

//void saveimage(Tcolor* img);

COLORREF colormap(double n);

void paint(HDC device, HBITMAP map);

double rand_num();

void saveimage(LPCSTR path, HBITMAP map);


HBITMAP render()
{
	srand(time(0));
	Tcolor img;
	double pixel;
	Tpair coor(0, 0);// x y coordinates of the pixel
	Tpair norx(xbounds._min, xbounds._max);//normalitation range
	Tpair nory(ybounds._min, ybounds._max);
	Tpair maxx(0, tamx);//max posible values of x
	Tpair maxy(0, tamy);


	for (int y = 0; y < tamy; y++)
	{
		for (int x = 0; x < tamx; x++)
		{
			pixel = 0;
			for (size_t i = 0; i < spp; i++)
			{
				coor.x = x + rand_num();
				coor.y = y + rand_num();
				coor = coor.map(norx, nory, maxx, maxy);
				pixel += converges(coor);
			}
			img.arr[tamx * y + x] = colormap(pixel);
		}
		if (y % 100 == 0)
		{
			cout << y << "\n";
		}
	}
	HBITMAP map = CreateBitmap(tamx, tamy, 1, 8 * 4, (void*)img.arr); // pointer to array
	return map;
}

//main "mandelbrot collision detection"
int converges(Tpair coor)
{
	int ret = it;
	double a = coor.x;
	double b = coor.y;
	double na = a;
	double nb = b;
	for (size_t i = 0; i < it; i++)
	{
		na = a * a - b * b;
		nb = 2 * a * b;
		a = na + coor.x;
		b = nb + coor.y;
		if (abs(a + b) > 16)
		{
			ret = i;
			break;
		}
	}
	return ret;
}

//converts an integer value into a greyscale tone
COLORREF colormap(double n)
{
	double grey_base = 255 * (n / it) * (1.0 / spp);
	return RGB(grey_base, grey_base, grey_base);
}

//Writes to file specified in ofstream (mildly optimized)
void saveimage(LPCSTR path, HBITMAP map)//only works with unidimensional color spaces
{
	HDC hdc = GetDC(NULL);
	SaveBitmapToFile(path, map);
}

//generates a random double in [0, 1)
inline double rand_num() {//esto es relento
	return 0;
}

void paint(HDC device, HBITMAP map)
{
	HDC src = CreateCompatibleDC(device);
	SelectObject(src, map);
	StretchBlt(device, 0, 0, tamx * rsize, tamy * rsize, src, 0, 0, tamx, tamy, SRCCOPY);
	//DeleteObject(map);
	DeleteDC(src); // Deleting temp HDC
}