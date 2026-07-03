#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <thread>
#include <chrono>
#include <random>
#include <functional>


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
            
            RECT saved_rect;
            bool is_fullscreen = false;

            int last_click_x = -1;
            int last_click_y = -1;
            int type_click = 0; 
            bool new_click = false;

            static LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
                if (uMsg == WM_NCCREATE) {
                    CREATESTRUCT* pCreate = reinterpret_cast<CREATESTRUCT*>(lParam);
                    ExternalScreen* pScreen = reinterpret_cast<ExternalScreen*>(pCreate->lpCreateParams);
                    SetWindowLongPtr(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(pScreen));
                }

                ExternalScreen* screen = reinterpret_cast<ExternalScreen*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));

                if (screen) {

                    if (uMsg == WM_SIZE) {

                        if (wParam == SIZE_MAXIMIZED && !screen->is_fullscreen) {
                            screen->ToggleFullScreen(true);
                            return 0;
                        }

                        if (wParam == SIZE_RESTORED && screen->is_fullscreen) {
                            screen->ToggleFullScreen(false);
                            return 0;
                        }

                        return DefWindowProc(hwnd, uMsg, wParam, lParam);
                    }

                    if (uMsg == WM_LBUTTONDOWN || uMsg == WM_RBUTTONDOWN || uMsg == WM_MBUTTONDOWN) {
                        screen->last_click_x = LOWORD(lParam);
                        screen->last_click_y = HIWORD(lParam);
                        screen->new_click = true;

                        if (uMsg == WM_LBUTTONDOWN) screen->type_click = 1;
                        if (uMsg == WM_RBUTTONDOWN) screen->type_click = 2;
                        if (uMsg == WM_MBUTTONDOWN) screen->type_click = 3;

                        return 0;
                    }
                }

                if (uMsg == WM_DESTROY) {
                    PostQuitMessage(0);
                    return 0;
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
                wc.lpfnWndProc = WindowProc;
                wc.hInstance = GetModuleHandle(NULL);
                wc.lpszClassName = "AI";
                wc.style = CS_HREDRAW | CS_VREDRAW; 
                RegisterClass(&wc);

                hwnd = CreateWindowEx(0, wc.lpszClassName, "AI", 
                    WS_OVERLAPPEDWINDOW | WS_VISIBLE, 
                    200, 100, w + 16, h + 39, NULL, NULL, wc.hInstance, this);
                

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

                pStretchDIBits(hdc, 0, 0, W, H, 0, 0, W, H, buffer.data(), &bmi, DIB_RGB_COLORS, SRCCOPY);

                MSG msg;
                while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
                    TranslateMessage(&msg);
                    DispatchMessage(&msg);
                    
                    if (msg.message == WM_QUIT) {
                        exit(0); 
                    }
                }
            } 
            void ToggleFullScreen(bool enable) {
                if (enable == is_fullscreen) return;

                if (enable) {
                    RECT rect;
                    GetWindowRect(hwnd, &rect);
                    saved_rect = rect;
                    int screenW = GetSystemMetrics(SM_CXSCREEN);
                    int screenH = GetSystemMetrics(SM_CYSCREEN);

                    W = screenW;
                    H = screenH;
                    buffer.resize(W * H, 0);

                    SetWindowLongPtr(hwnd, GWL_STYLE, WS_POPUP | WS_VISIBLE);
                    SetWindowPos(hwnd, HWND_TOP, 0, 0, W, H, SWP_FRAMECHANGED);
                } else {
                    SetWindowLongPtr(hwnd, GWL_STYLE, WS_OVERLAPPEDWINDOW | WS_VISIBLE);
                    SetWindowPos(hwnd, HWND_NOTOPMOST, saved_rect.left, saved_rect.top, 
                                saved_rect.right - saved_rect.left, 
                                saved_rect.bottom - saved_rect.top, SWP_FRAMECHANGED);
                    
                    W = saved_rect.right - saved_rect.left;
                    H = saved_rect.bottom - saved_rect.top;
                    buffer.resize(W * H, 0);
                }

                is_fullscreen = enable;
            }
            void cls() {
                std::fill(buffer.begin(), buffer.end(), 0);
            }

            void fillScreen(unsigned char r, unsigned char g, unsigned char b) {
                unsigned int color = (r << 16) | (g << 8) | b;
                std::fill(buffer.begin(), buffer.end(), color);
            }
            void putPixel(int x, int y, unsigned char r, unsigned char g, unsigned char b) {
                if (x >= 0 && x < W && y >= 0 && y < H) {
                    buffer[y * W + x] = (r << 16) | (g << 8) | b;
                }
            }
    };
}