#include <algo/blast/gpu_blast/gpu_blast_multi_gpu_utils.hpp>
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <thread>

static int discoveries = 0;
static int devices = 2;
static int destroyed = 0;
static thread_local int selected = -1;
cudaError_t cudaGetDeviceCount(int* n) { ++discoveries; *n = devices; return cudaSuccess; }
cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p, int) {
    p->major = 11; p->minor = 0; std::strcpy(p->name, "mock Thor"); return cudaSuccess;
}
cudaError_t cudaSetDevice(int d) { selected = d; return cudaSuccess; }
cudaError_t cudaFree(void*) { return cudaSuccess; }
const char* cudaGetErrorString(cudaError_t) { return "mock CUDA error"; }
struct Resource : GpuObject { ~Resource() { ++destroyed; } };

int main() {
    assert(discoveries == 0); // No CUDA runtime call during global construction.
    assert(BlastMGPUUtil.InitGPUs(false, -1) == 0);
    assert(discoveries == 0);
    int id = 123;
    BlastMGPUUtil.ThreadFetchGPU(id);
    assert(id == -1);
    devices = 0;
    bool failed = false;
    try { BlastMGPUUtil.InitGPUs(true, -1); } catch (const std::runtime_error&) { failed = true; }
    assert(failed);
    devices = 2;
    assert(BlastMGPUUtil.InitGPUs(true, 1) == 1);
    BlastMGPUUtil.ThreadFetchGPU(id);
    assert(id == 1 && selected == 1);
    id = -42;
    BlastMGPUUtil.ThreadFetchGPU(id);
    assert(id == 1); // Regression: this used to return the thread map key.
    std::thread competing([] {
        int other = 5;
        BlastMGPUUtil.ThreadFetchGPU(other);
        assert(other == -1);
    });
    competing.join();
    GpuHandle* handle = BlastMGPUUtil.GetCurrentGPUHandle();
    assert(handle);
    handle->Data.m_global = new Resource();
    handle->Data.m_local = new Resource();
    // Cleanup must cover outstanding leases, not just idle devices.
    BlastMGPUUtil.ReleaseGPUs();
    assert(destroyed == 2);
    BlastMGPUUtil.ReleaseGPUs();
    assert(destroyed == 2);
    assert(BlastMGPUUtil.InitGPUs(true, 0) == 1);
    BlastMGPUUtil.ThreadFetchGPU(id);
    assert(id == 0);
    BlastMGPUUtil.ThreadReplaceGPU();
    assert(!BlastMGPUUtil.GetCurrentGPUHandle());
    BlastMGPUUtil.ThreadFetchGPU(id);
    assert(id == 0);
    BlastMGPUUtil.ThreadReplaceGPU();
    BlastMGPUUtil.ReleaseGPUs();
    failed = false;
    try { BlastMGPUUtil.InitGPUs(true, 9); } catch (const std::runtime_error&) { failed = true; }
    assert(failed);
    assert(!BlastMGPUUtil.b_useGpu);
}
