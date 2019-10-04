
// CUDA runtime
#include <iostream>
#include <algorithm>

//////////////////////////////////////////////////////////////////////////
#include <algo/blast/gpu_blast/gpu_blast_multi_gpu_utils.hpp>
#include <algo/blast/gpu_blast/thread_work_queue.hpp>


GpuBlastMultiGPUsUtils::GpuBlastMultiGPUsUtils()
{
	b_useGpu = false;
	i_num_limited = 0;
	select_id = -1;

	checkCudaErrors(cudaGetDeviceCount(&i_GPU_N));
	for (int i = 0; i < i_GPU_N; i++)
	{
		cudaDeviceProp deviceProp;
		checkCudaErrors(cudaGetDeviceProperties(&deviceProp, i));
		int version = deviceProp.major * 10 + deviceProp.minor;
		if (version >= 13)
		{
			GpuHandle * gpu_handle = new GpuHandle();
			gpu_handle->Prop = deviceProp;
			gpu_handle->InUsed = false;
			gpu_handle->Data.m_global = NULL;
			gpu_handle->Data.m_local = NULL;
			mt_GPU.insert (GpuHandleMapPairType(i, gpu_handle));
			q_gpu_ids.push_back(i);
		}
	}

	i_GPU_N = q_gpu_ids.size();
}

GpuBlastMultiGPUsUtils::~GpuBlastMultiGPUsUtils()
{
	ReleaseGPUs();
}

int GpuBlastMultiGPUsUtils::InitGPUs(bool use_gpu, int gpu_id)
{
	if (use_gpu == false) 
		return 0;

	if (i_GPU_N < 1) 
	{
		cout << "There is no GPU card compute capability > 1.3."
		     << "It runs in CPU mode." 
			 << endl;
		return 0;
	}

	if (gpu_id != -1)
	{
		if(mt_GPU.find(gpu_id) == mt_GPU.end())
		{
			cout << "Please choose GPU card compute capability > 1.3." << endl;
			return 0;
		}
		b_useGpu = true;
		select_id = gpu_id;
		return 1;
	}
	b_useGpu = true;
	return i_GPU_N;
}

void GpuBlastMultiGPUsUtils::ReleaseGPUs()
{
	if (b_useGpu)
	{
		while(!q_gpu_ids.empty())
		{
			int gpu_id = q_gpu_ids.back();
			checkCudaErrors(cudaSetDevice(gpu_id));

			GpuHandleMapType::iterator itr = mt_GPU.find(gpu_id);
			if (itr != mt_GPU.end())
			{
				GpuHandle* gpu_handle = itr->second;

				if (gpu_handle != NULL)
				{
					if (gpu_handle->Data.m_global != NULL)
					{
						delete gpu_handle->Data.m_global;
					}
					if (gpu_handle->Data.m_local != NULL)
					{
						delete gpu_handle->Data.m_local;
					}
					delete gpu_handle;
				}
			}

			cudaDeviceReset();
			q_gpu_ids.pop_back();
		}
	}
}

void GpuBlastMultiGPUsUtils::ThreadFetchGPU(int & gpu_id)
{
	mt_lock.SectionLock();
	if (!b_useGpu)
	{
		gpu_id = -1;
	}
	else
	{
		unsigned long p_thread_id = mt_lock.GetCurrentThreadID();
		if (mt_threads.find(p_thread_id) != mt_threads.end())
		{
			gpu_id = mt_threads.find(p_thread_id)->first;
		}
		else
		{
			if (q_gpu_ids.size() > 0)
			{
				gpu_id = q_gpu_ids.back();
				mt_threads.insert(ThreadGPUPairType(p_thread_id, gpu_id));
				checkCudaErrors(cudaSetDevice(gpu_id));
				string thread_name = "GPU thread ";
				thread_name += gpu_id;
				mt_lock.SetCurrentThreadName(p_thread_id, thread_name);
				q_gpu_ids.pop_back();
			}
			else
			{
				gpu_id = -1;
			}
		}
		//cout <<"Thread:"<< p_thread_id <<" GPU id:"<< gpu_id << endl;
	}
	mt_lock.SectionUnlock();
}

void GpuBlastMultiGPUsUtils::ThreadReplaceGPU()
{
	mt_lock.SectionLock();

	unsigned long p_thread_id = mt_lock.GetCurrentThreadID();
	ThreadGPUMapType::iterator itr = mt_threads.find(p_thread_id);
	if( itr != mt_threads.end())
	{	
		int gpu_id = itr->second;
		q_gpu_ids.push_back(gpu_id);
		mt_threads.erase(itr);
	}
	mt_lock.SectionUnlock();
}
//////////////////////////////////////////////////////////////////////////

GpuHandle* GpuBlastMultiGPUsUtils::GetCurrentGPUHandle()
{
	unsigned long p_thread_id = mt_lock.GetCurrentThreadID();

	//cout << "thread Id: " << p_thread_id << endl;

	if(mt_threads.find(p_thread_id) != mt_threads.end())
	{
		int gpu_id = mt_threads[p_thread_id];
		checkCudaErrors(cudaSetDevice(gpu_id));
		return mt_GPU[gpu_id];
	}

	return NULL;
}

GpuBlastMultiGPUsUtils& BlastMGPUUtil = GpuBlastMultiGPUsUtils::instance();


int Blast_gpu_Init(bool isInit, int gpu_id)
{
    return BlastMGPUUtil.InitGPUs(isInit, gpu_id);  
}
void Blast_gpu_Release()
{
    BlastMGPUUtil.ReleaseGPUs();
}

