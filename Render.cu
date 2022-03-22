#include "Render.h"

__global__ void renderPixel(const Tpair& norx, const Tpair& nory, const Tpair& maxx, const Tpair& maxy, int it, Tcolor* d_img)
{
	double pixel = 0;
	Tpair coor(0, 0);
	int x = threadIdx.x;
	int y = blockDim.x;
	for (size_t i = 0; i < spp; i++)
	{
		coor.x = x;
		coor.y = y;
		coor.map(norx, nory, maxx, maxy);
		pixel += converges(coor, it);
	}
	d_img->arr[tamx * y + x] = colormap(pixel, it);
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


HBITMAP render(Tbounds xbounds, Tbounds ybounds, int it)
{
	srand(time(0));
	Tcolor* img;
	Tcolor* d_img;
	size_t bytes = (tamx * tamy) * sizeof(COLORREF);
	double pixel;
	cudaMalloc(&d_img, bytes);
	Tpair coor(0, 0);// x y coordinates of the pixel
	Tpair norx(xbounds._min, xbounds._max);//normalitation range
	Tpair nory(ybounds._min, ybounds._max);
	Tpair maxx(0, tamx);//max posible values of x
	Tpair maxy(0, tamy);
	dim3 threadspb(1024, 0, 0);
	dim3 blockspf(1024, 0, 0);
	renderPixel <<<blockspf, threadspb >>> (norx, nory, maxx, maxy, it, d_img);
	cudaMemcpy(img, d_img, bytes, cudaMemcpyDeviceToHost);
	HBITMAP map = CreateBitmap(tamx, tamy, 1, 8 * 4, (void*)img->arr); // pointer to array
	return map;
}