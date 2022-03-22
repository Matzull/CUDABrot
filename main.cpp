// HelloWindowsDesktop.cpp
// compile with: /D_UNICODE /DUNICODE /DWIN32 /D_WINDOWS /c

#include <windows.h>
#include <stdlib.h>
#include <string.h>
#include <tchar.h>
#include "Render.h"
#include <chrono>
using namespace std::chrono;

// Global variables
void render_frame(HWND hWnd);
void benchmark(HWND hWnd);

Tbounds xbounds;
Tbounds ybounds;
int it = 10;//iterations to calculate
// The main window class name.
static TCHAR szWindowClass[] = _T("DesktopApp");

// The string that appears in the application's title bar.
static TCHAR szTitle[] = _T("Windows Desktop Guided Tour Application");

// Stored instance handle for use in Win32 API calls such as FindResource
HINSTANCE hInst;

HBITMAP map;

// Forward declarations of functions included in this code module:
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

int WINAPI WinMain(
    _In_ HINSTANCE hInstance,
    _In_opt_ HINSTANCE hPrevInstance,
    _In_ LPSTR     lpCmdLine,
    _In_ int       nCmdShow
)
{
    WNDCLASSEX wcex;

    wcex.cbSize = sizeof(WNDCLASSEX);
    wcex.style = CS_HREDRAW | CS_VREDRAW;
    wcex.lpfnWndProc = WndProc;
    wcex.cbClsExtra = 0;
    wcex.cbWndExtra = 0;
    wcex.hInstance = hInstance;
    wcex.hIcon = LoadIcon(wcex.hInstance, IDI_APPLICATION);
    wcex.hCursor = LoadCursor(NULL, IDC_ARROW);
    wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wcex.lpszMenuName = NULL;
    wcex.lpszClassName = szWindowClass;
    wcex.hIconSm = LoadIcon(wcex.hInstance, IDI_APPLICATION);

    if (!RegisterClassEx(&wcex))
    {
        MessageBox(NULL,
            _T("Call to RegisterClassEx failed!"),
            _T("Windows Desktop Guided Tour"),
            NULL);

        return 1;
    }

    // Store instance handle in our global variable
    hInst = hInstance;

    //HWND hWnd = CreateWindowW(szWindowClass, szTitle, WS_OVERLAPPEDWINDOW, (pantx - tamx * scale) / 2, (panty - tamy * scale) / 2, tamx * scale, tamy * scale, nullptr, nullptr, hInstance, nullptr);
    // The parameters to CreateWindowEx explained:
    // WS_EX_OVERLAPPEDWINDOW : An optional extended window style.
    // szWindowClass: the name of the application
    // szTitle: the text that appears in the title bar
    // WS_OVERLAPPEDWINDOW: the type of window to create
    // CW_USEDEFAULT, CW_USEDEFAULT: initial position (x, y)
    // 500, 100: initial size (width, length)
    // NULL: the parent of this window
    // NULL: this application does not have a menu bar
    // hInstance: the first parameter from WinMain
    // NULL: not used in this application
    int pantx = GetSystemMetrics(0);
    int panty = GetSystemMetrics(1);

    HWND hWnd = CreateWindowEx(
        WS_EX_OVERLAPPEDWINDOW,
        szWindowClass,
        szTitle,
        WS_OVERLAPPEDWINDOW,
        (pantx - tamx * rsize) / 2, (panty - tamy * rsize) / 2,
        tamx * rsize, tamy * rsize,
        NULL,
        NULL,
        hInstance,
        NULL
    );
    render_frame(hWnd);
    if (!hWnd)
    {
        MessageBox(NULL,
            _T("Call to CreateWindow failed!"),
            _T("Windows Desktop Guided Tour"),
            NULL);

        return 1;
    }

    // The parameters to ShowWindow explained:
    // hWnd: the value returned from CreateWindow
    // nCmdShow: the fourth parameter from WinMain
    ShowWindow(hWnd,nCmdShow);
    UpdateWindow(hWnd);

    // Main message loop:
    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0))
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    return (int)msg.wParam;
}

//  FUNCTION: WndProc(HWND, UINT, WPARAM, LPARAM)
//
//  PURPOSE:  Processes messages for the main window.
//
//  WM_PAINT    - Paint the main window
//  WM_DESTROY  - post a quit message and return
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    //Code:
    //arrow-up (+y)
    //arrow-down (-y)
    //arrow-left (-x)
    //arrow-right (+x)
    //+ (zoom)
    //0 (-it)
    //1 (+it)
    //2 (save)

    PAINTSTRUCT ps;
    HDC hdc;
    TCHAR greeting[] = _T("Hello, Windows desktop!");

    switch (message)
    {
    case WM_PAINT:
        hdc = BeginPaint(hWnd, &ps);
        //HDC hdc = GetDC(msg.hwnd);
        paint(hdc, map);
        // Here your application is laid out./*GetMessage(&msg, nullptr, 0, 0);
        EndPaint(hWnd, &ps);
        break;
    case WM_DESTROY:
        PostQuitMessage(0);
        break;
    case WM_KEYDOWN:
        switch (wParam)
        {
        case VK_NUMPAD1:
            it += 5;
            render_frame(hWnd);
            break;

        case VK_NUMPAD0:
            it -= (it == 0) ? 0 : 5;
            render_frame(hWnd);
            break;

        case VK_NUMPAD2:
            saveimage("out.bmp", map);
            break;

        case VK_UP:
            ybounds.move(-0.25);
            render_frame(hWnd);
            break;

        case VK_DOWN:
            ybounds.move(0.25);
            render_frame(hWnd);
            break;

        case VK_LEFT:
            xbounds.move(-0.25);
            render_frame(hWnd);
            break;

        case VK_RIGHT:
            xbounds.move(0.25);
            render_frame(hWnd);
            break;

        case VK_ADD:
            xbounds.scale(1.2);
            ybounds.scale(1.2);
            render_frame(hWnd);
            break;

        case VK_SUBTRACT:
            xbounds.scale(0.8);
            ybounds.scale(0.8);
            render_frame(hWnd);
            break;

        case VK_NUMPAD3:
            benchmark(hWnd);
            break;

        default:
            break;
        }
    default:
        return DefWindowProc(hWnd, message, wParam, lParam);
        break;
    }

    return 0;
}
void benchmark(HWND hWnd)
{
    xbounds.move(-0.25);
    ybounds.move(-0.25);
    xbounds.scale(2);
    ybounds.scale(2);
    int oldit = it;
    it = 10000;
    auto start = high_resolution_clock::now();
    render_frame(hWnd);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    int dur = duration.count();
    it = oldit;
    xbounds.move(0.25);
    ybounds.move(0.25);
    xbounds.scale(0.5);
    ybounds.scale(0.5);
}

void render_frame(HWND hWnd)
{
    HDC hdc;
    map = render(xbounds, ybounds, it);
    hdc = GetDC(hWnd);
    paint(hdc, map);
}