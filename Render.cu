#include "Renderc.h"
#include "curand.h"
int converges(Tpair coor, int it);
COLORREF colormap(double n, int it);
void mapper(const Tpair& norx, const Tpair& nory, const Tpair& maxx, const Tpair& maxy, Tpair& coor);

__global__ void renderPixel(Tpair* norx, Tpair* nory, Tpair* maxx, Tpair* maxy, int it, COLORREF* d_img)
{
	double pixel = 0;
	Tpair coor;
	coor.fill(0, 0);
	int tid = threadIdx.x;
	int bdim = blockDim.x;
	int bid = blockIdx.x;
	double sppx[4] = { 0.25, 0.75, 0.25, 0.75 };
	double sppy[4] = { 0.25, 0.25, 0.75, 0.25 };

	for (size_t i = 0; i < 4; i++)
	{
		coor.x = tid + sppx[i];
		coor.y = bid + sppy[i];
		mapper(*norx, *nory, *maxx, *maxy, coor);
		pixel += converges(coor, it);
	}
	d_img[blockDim.x * blockIdx.x + threadIdx.x] = colormap(pixel, it);
}

//__device__ void mapper(Tpair* norx, Tpair* nory, Tpair* maxx, Tpair* maxy, Tpair* coor)//Normalizes pair this with max value max to interval nor
//{
//	//double ot = (coor->x - maxx->x) / (maxx->y - maxx->x);
//	coor->x = (((coor->x - maxx->x) * (norx->y - norx->x)) / (maxx->y - maxx->x)) + norx->x;
//	coor->y = (((coor->y - maxy->x) * (nory->y - nory->x)) / (maxy->y - maxy->x)) + nory->x;
//}

__device__ void mapper(const Tpair& norx, const Tpair& nory, const Tpair& maxx, const Tpair& maxy, Tpair& coor)//Normalizes pair this with max value max to interval nor
{
	//double ot = (coor->x - maxx->x) / (maxx->y - maxx->x);
	coor.x = (((coor.x - maxx.x) * (norx.y - norx.x)) / (maxx.y - maxx.x)) + norx.x;
	coor.y = (((coor.y - maxy.x) * (nory.y - nory.x)) / (maxy.y - maxy.x)) + nory.x;
}

//main "mandelbrot collision detection"
__device__ int converges(Tpair coor, int it)
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
__device__ COLORREF colormap(double n, int it)
{
	double nor = n * 0.25 / it;
	double grey_base = 255 * nor * nor;
	grey_base = (n * 0.25 == it) ? 0 : grey_base;
	return RGB(grey_base / nor, grey_base, grey_base);
}

//Writes to file specified in ofstream (mildly optimized)
void saveimage(LPCSTR path, HBITMAP map)//only works with unidimensional color spaces
{
	HDC hdc = GetDC(NULL);
	SaveBitmapToFile(path, map);
}

//generates a random double in [0, 1)
__device__ double rand_num() {//esto es relento
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


HBITMAP render(Tbounds xbounds, Tbounds ybounds, int it)
{
	srand(time(0));
	//Tcolor img;
	size_t bytes = (tamx * tamy) * sizeof(COLORREF);
	COLORREF* img;
	COLORREF* d_img;	
	img = (COLORREF*)malloc(bytes);
	cudaMalloc((void**)&d_img, bytes);

	Tpair norx;
	norx.fill(xbounds._min, xbounds._max);
	Tpair* d_norx;//normalitation range
	cudaMalloc((void**)&d_norx, sizeof(Tpair));
	cudaMemcpy(d_norx, &norx, sizeof(Tpair), cudaMemcpyHostToDevice);

	Tpair nory;
	nory.fill(ybounds._min, ybounds._max);
	Tpair* d_nory;//normalitation range
	cudaMalloc((void**)&d_nory, sizeof(Tpair));
	cudaMemcpy(d_nory, &nory, sizeof(Tpair), cudaMemcpyHostToDevice);

	Tpair maxx;
	maxx.fill(0, tamx);
	Tpair* d_maxx;//normalitation range
	cudaMalloc((void**)&d_maxx, sizeof(Tpair));
	cudaMemcpy(d_maxx, &maxx, sizeof(Tpair), cudaMemcpyHostToDevice);

	Tpair maxy;
	maxy.fill(0, tamy);
	Tpair* d_maxy;//normalitation range
	cudaMalloc((void**)&d_maxy, sizeof(Tpair));
	cudaMemcpy(d_maxy, &maxy, sizeof(Tpair), cudaMemcpyHostToDevice);


	/*	dim3 threadspb(0, 1024, 0);
	dim3 blockspf(0, 1024, 0);*/

	renderPixel <<<1024, 1024>>> (d_norx, d_nory, d_maxx, d_maxy, it, d_img);

	cudaMemcpy(img, d_img, bytes, cudaMemcpyDeviceToHost);
	
	HBITMAP map = CreateBitmap(tamx, tamy, 1, 8 * 4, (void*)img); // pointer to array
	free(img);
	cudaFree(d_img);
	cudaFree(d_norx);
	cudaFree(d_nory);
	cudaFree(d_maxx);
	cudaFree(d_maxy);
	return map;
}