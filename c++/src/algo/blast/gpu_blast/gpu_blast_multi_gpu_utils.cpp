// CUDA discovery is deliberately lazy: linked database utilities, -help,
// -version and -use_gpu false must not require an accessible NVIDIA driver.
#include <algo/blast/gpu_blast/gpu_blast_multi_gpu_utils.hpp>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace {
void CheckCuda(cudaError_t status, const char* operation)
{
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("G-BLASTN: ") + operation + ": " +
                                 cudaGetErrorString(status));
}
}

GpuBlastMultiGPUsUtils::GpuBlastMultiGPUsUtils()
    : b_useGpu(false), i_GPU_N(0), select_id(-1), i_num_limited(0) {}

GpuBlastMultiGPUsUtils::~GpuBlastMultiGPUsUtils() { ReleaseGPUs(); }

int GpuBlastMultiGPUsUtils::InitGPUs(bool use_gpu, int gpu_id)
{
    ReleaseGPUs();
    if (!use_gpu) {
        std::cerr << "GBLASTN_BACKEND backend=cpu\n";
        return 0;
    }
    std::lock_guard<std::mutex> guard(mt_lock);
    CheckCuda(cudaGetDeviceCount(&i_GPU_N), "GPU requested; CUDA discovery failed");
    if (i_GPU_N == 0)
        throw std::runtime_error("G-BLASTN: GPU requested but no CUDA device is available; use -use_gpu false for explicit CPU execution");
    if (gpu_id < -1 || gpu_id >= i_GPU_N)
        throw std::runtime_error("G-BLASTN: -gpu_id does not identify an available CUDA device");
    for (int i = 0; i < i_GPU_N; ++i) {
        if (gpu_id != -1 && i != gpu_id) continue;
        cudaDeviceProp prop;
        CheckCuda(cudaGetDeviceProperties(&prop, i), "cudaGetDeviceProperties");
        CheckCuda(cudaSetDevice(i), "cudaSetDevice");
        // Establish the context here so initialization failures cannot turn
        // into a silently successful CPU search later.
        CheckCuda(cudaFree(0), "CUDA context initialization");
        std::unique_ptr<GpuHandle> handle(new GpuHandle());
        handle->Prop = prop;
        handle->InUsed = false;
        handle->Data.m_global = NULL;
        handle->Data.m_local = NULL;
        mt_GPU.insert(GpuHandleMapPairType(i, handle.get()));
        handle.release();
        q_gpu_ids.push_back(i);
        std::cerr << "GBLASTN_DEVICE backend=cuda id=" << i
                  << " name=\"" << prop.name << "\" compute_capability=" << prop.major
                  << '.' << prop.minor << "\n";
    }
    select_id = gpu_id;
    b_useGpu = true;
    return static_cast<int>(q_gpu_ids.size());
}

void GpuBlastMultiGPUsUtils::ReleaseGPUs()
{
    std::lock_guard<std::mutex> guard(mt_lock);
    // The owner joins workers before calling this method. Iterate the handle
    // map, not the free-device pool: a failed query may still own a lease.
    for (GpuHandleMapType::iterator i = mt_GPU.begin(); i != mt_GPU.end(); ++i) {
        const cudaError_t status = cudaSetDevice(i->first);
        if (status != cudaSuccess)
            std::cerr << "G-BLASTN: device cleanup: " << cudaGetErrorString(status) << '\n';
        delete i->second->Data.m_global;
        delete i->second->Data.m_local;
        delete i->second;
    }
    mt_GPU.clear();
    mt_threads.clear();
    q_gpu_ids.clear();
    b_useGpu = false;
    i_GPU_N = 0;
    select_id = -1;
    // Do not cudaDeviceReset(): ownership is limited to our allocations.
}

void GpuBlastMultiGPUsUtils::ThreadFetchGPU(int& gpu_id)
{
    std::lock_guard<std::mutex> guard(mt_lock);
    gpu_id = -1;
    if (!b_useGpu) return;
    const std::thread::id thread_id = std::this_thread::get_id();
    ThreadGPUMapType::iterator found = mt_threads.find(thread_id);
    if (found != mt_threads.end()) {
        gpu_id = found->second; // GPU-ID value, never the thread-ID map key.
        CheckCuda(cudaSetDevice(gpu_id), "cudaSetDevice for existing lease");
    } else if (!q_gpu_ids.empty()) {
        gpu_id = q_gpu_ids.back();
        CheckCuda(cudaSetDevice(gpu_id), "cudaSetDevice for new lease");
        mt_threads.insert(ThreadGPUPairType(thread_id, gpu_id));
        q_gpu_ids.pop_back();
    }
}

void GpuBlastMultiGPUsUtils::ThreadReplaceGPU()
{
    std::lock_guard<std::mutex> guard(mt_lock);
    ThreadGPUMapType::iterator found = mt_threads.find(std::this_thread::get_id());
    if (found != mt_threads.end()) {
        q_gpu_ids.push_back(found->second);
        mt_threads.erase(found);
    }
}

GpuHandle* GpuBlastMultiGPUsUtils::GetCurrentGPUHandle()
{
    std::lock_guard<std::mutex> guard(mt_lock);
    ThreadGPUMapType::iterator found = mt_threads.find(std::this_thread::get_id());
    if (found == mt_threads.end()) return NULL;
    CheckCuda(cudaSetDevice(found->second), "cudaSetDevice for query");
    return mt_GPU.find(found->second)->second;
}

GpuBlastMultiGPUsUtils& BlastMGPUUtil = GpuBlastMultiGPUsUtils::instance();

int Blast_gpu_Init(bool use_gpu, int gpu_id)
{
    return BlastMGPUUtil.InitGPUs(use_gpu, gpu_id);
}
void Blast_gpu_Release() { BlastMGPUUtil.ReleaseGPUs(); }
