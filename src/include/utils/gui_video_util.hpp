#include <iostream>
#include <vector>
#include <cmath>
#include <thread>
#include <chrono>
#include <random>

#include <windows.h>
using namespace std;

namespace GUI_HUD
{

    typedef int(WINAPI* PFN_StretchDIBits)(HDC, int, int, int, int, int, int, int, int, const VOID*, const BITMAPINFO*, UINT, DWORD);

    class ExternalScreen {

        public:

            //? Windows APIS

            HWND hwnd;
            HDC hdc;
            int W, H;
            vector<unsigned int> buffer;
            PFN_StretchDIBits pStretchDIBits = nullptr; 
            HMODULE hGdi32 = nullptr;

            int last_click_x = -1;
            int last_click_y = -1;
            int type_click = 0; 
            bool new_click = false;

            static LRESULT CALLBACK WindowProcCustom(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {

                ExternalScreen* screen = reinterpret_cast<ExternalScreen*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));

                if (screen) {
                    if (uMsg == WM_LBUTTONDOWN || uMsg == WM_RBUTTONDOWN || uMsg == WM_MBUTTONDOWN) {
                        screen->last_click_x = LOWORD(lParam);
                        screen->last_click_y = HIWORD(lParam);
                        screen->new_click = true;

                        if (uMsg == WM_LBUTTONDOWN)  screen->type_click = 1;
                        if (uMsg == WM_RBUTTONDOWN)  screen->type_click = 2;
                        if (uMsg == WM_MBUTTONDOWN)  screen->type_click = 3;

                        return 0;
                    }
                    if (uMsg == WM_DESTROY) {
                        PostQuitMessage(0);
                        return 0;
                    }
                }
                return DefWindowProc(hwnd, uMsg, wParam, lParam);
            }

            //? Windows Functions

            ExternalScreen(int w, int h) : W(w), H(h) {

                buffer.resize(w * h, 0);
                hGdi32 = LoadLibraryA("gdi32.dll");
                if (hGdi32) {
                    pStretchDIBits = (PFN_StretchDIBits)GetProcAddress(hGdi32, "StretchDIBits");
                }

                WNDCLASS wc = {};
                wc.lpfnWndProc = WindowProcCustom;
                wc.hInstance = GetModuleHandle(NULL);
                wc.lpszClassName = "AI";
                RegisterClass(&wc);

                hwnd = CreateWindowEx(0, wc.lpszClassName, "AI", 
                    WS_OVERLAPPEDWINDOW | WS_VISIBLE, 
                    200, 100, w + 16, h + 39, NULL, NULL, wc.hInstance, NULL);
                
                SetWindowLongPtr(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(this));

                hdc = GetDC(hwnd);
            }
            ~ExternalScreen() {
                ReleaseDC(hwnd, hdc);
                DestroyWindow(hwnd);
                if (hGdi32) FreeLibrary(hGdi32);
            }
            void reloadScreen() {
                if (!pStretchDIBits) return;

                BITMAPINFO bmi = {};
                bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
                bmi.bmiHeader.biWidth = W;
                bmi.bmiHeader.biHeight = -H; 
                bmi.bmiHeader.biPlanes = 1;
                bmi.bmiHeader.biBitCount = 32;
                bmi.bmiHeader.biCompression = BI_RGB;

                // Apelăm funcția prin pointerul obținut din DLL (Linkerul g++ habar n-are!)
                pStretchDIBits(hdc, 0, 0, W, H, 0, 0, W, H, buffer.data(), &bmi, DIB_RGB_COLORS, SRCCOPY);

                MSG msg;
                while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
                    TranslateMessage(&msg);
                    DispatchMessage(&msg);
                }
            }
            
            //?


            //* Display Functions

            void putPixel(int x, int y, unsigned char r, unsigned char g, unsigned char b) {
                if (x >= 0 && x < W && y >= 0 && y < H) {
                    buffer[y * W + x] = (r << 16) | (g << 8) | b;
                }
            }


        };

}