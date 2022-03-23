#include "Render.h"
#include <thread>
#include <mutex>
#include <vector>

mutex myMutex;

void renderThread(const Tpair& norx, const Tpair& nory, const Tpair& maxx, const Tpair& maxy, int it, int threads, int index, Tcolor &img);

/*
HBITMAP render(Tbounds xbounds, Tbounds ybounds, int it)
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
				coor.map(norx, nory, maxx, maxy);
				pixel += converges(coor, it);
			}
			img.arr[tamx * y + x] = colormap(pixel, it);
		}
		if (y % 100 == 0)
		{
			cout << y << "\n";
		}
	}
	HBITMAP map = CreateBitmap(tamx, tamy, 1, 8 * 4, (void*)img.arr); // pointer to array
	return map;
}
*/

HBITMAP render(Tbounds xbounds, Tbounds ybounds, int it)
{
	srand(time(0));
	Tcolor img;
	Tpair coor(0, 0);// x y coordinates of the pixel
	Tpair norx(xbounds._min, xbounds._max);//normalitation range
	Tpair nory(ybounds._min, ybounds._max);
	Tpair maxx(0, tamx);//max posible values of x
	Tpair maxy(0, tamy);
	int threadcount = thread::hardware_concurrency() - 2;
	
	vector<thread> renderers;

	for (int i = 0; i < threadcount; i++)
	{
		renderers.push_back(thread(renderThread, ref(norx), ref(nory), ref(maxx), ref(maxy), it, threadcount, i, ref(img)));
	}
	for (auto& t : renderers)
	{
		t.join();
	}
	HBITMAP map = CreateBitmap(tamx, tamy, 1, 8 * 4, (void*)img.arr); // pointer to array
	return map;
}

void renderThread(const Tpair& norx, const Tpair& nory, const Tpair& maxx, const Tpair& maxy, int it, int threads, int index, Tcolor &img)
{
	double pixel = 0;
	Tpair coor(0, 0);
	for (size_t y = 0 + index; y < tamy; y += threads)
	{
		for (size_t x = 0; x < tamx; x++)
		{
			for (size_t i = 0; i < spp; i++)
			{
				coor.x = x + rand_num();
				coor.y = y + rand_num();
				coor.map(norx, nory, maxx, maxy);
				pixel += converges(coor, it);
			}
			COLORREF insert = colormap(pixel, it);
			//myMutex.lock();
			img.arr[tamx * y + x] = insert;
			//myMutex.unlock();
			pixel = 0;
		}
	}
}

//main "mandelbrot collision detection"
int converges(Tpair coor, int it)
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
COLORREF colormap(double n, int it)
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
	return rand() / (RAND_MAX + 1.0);
}

void paint(HDC device, HBITMAP map)
{
	HDC src = CreateCompatibleDC(device);
	SelectObject(src, map);
	StretchBlt(device, 0, 0, tamx * rsize, tamy * rsize, src, 0, 0, tamx, tamy, SRCCOPY);
	//DeleteObject(map);
	DeleteDC(src); // Deleting temp HDC
}