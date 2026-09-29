// Test-only CUDA runtime surface. Never included by a production build.
#pragma once
#include <stddef.h>
struct cudaDeviceProp { int major; int minor; char name[64]; };
enum cudaError_t { cudaSuccess = 0, cudaErrorNoDevice = 100 };
cudaError_t cudaGetDeviceCount(int*);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp*, int);
cudaError_t cudaSetDevice(int);
cudaError_t cudaFree(void*);
const char* cudaGetErrorString(cudaError_t);
