// Optional environment diagnostic; this is NOT BLAST kernel acceptance evidence.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#define CHECK(call) do { cudaError_t s = (call); if (s != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(s)); return 1; } } while (0)
__global__ void probe(int* result) { result[0] = 110; }
int main() {
    int devices = 0, driver = 0, runtime = 0;
    CHECK(cudaGetDeviceCount(&devices));
    if (!devices) return 1;
    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, 0));
    CHECK(cudaDriverGetVersion(&driver)); CHECK(cudaRuntimeGetVersion(&runtime));
    std::printf("device=%s capability=%d.%d driver=%d runtime=%d integrated=%d\n",
                p.name, p.major, p.minor, driver, runtime, p.integrated);
    if (p.major != 11 || p.minor != 0) return 2;
    int* d = NULL; int h = 0;
    CHECK(cudaMalloc(&d, sizeof(int)));
    probe<<<1,1>>>(d); CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(&h, d, sizeof(h), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d));
    if (h != 110) return 3;
    std::puts("Thor environment probe completed; BLAST acceptance still required.");
    return 0;
}
