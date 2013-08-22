#include <algo/blast/gpu_blast/gpu_blast_multi_gpu_utils.hpp>
#include <algo/blast/gpu_blast/thread_work_queue.hpp>
// CUDA runtime
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <iostream>
//////////////////////////////////////////////////////////////////////////



GpuBlastMultiGPUsUtils::GpuBlastMultiGPUsUtils()
{
	b_useGpu = false;
	i_num_limited = 0;
}

GpuBlastMultiGPUsUtils::~GpuBlastMultiGPUsUtils()
{
	ReleaseGPUs();
}

int GpuBlastMultiGPUsUtils::InitGPUs(bool use_gpu, int gpu_id)
{
	if (b_useGpu == true) return i_GPU_N;
	b_useGpu = use_gpu;
	if (b_useGpu == false) 
		return 0;

	checkCudaErrors(cudaGetDeviceCount(&i_GPU_N));
	if ((i_GPU_N < 1) || (gpu_id > i_GPU_N-1)) 
		return 0;

	if (gpu_id != -1)
	{
		q_gpu_ids.push(gpu_id);
		GpuData* gpu_data = new GpuData();
		gpu_data->m_global = NULL;
		gpu_data->m_local = NULL;
		m_GpuData.insert(GPUDataMapPairType(gpu_id, gpu_data));
		return 1;
	}
	else
	{
		for ( int i = 0; i < i_GPU_N; i++)
		{								 
			q_gpu_ids.push(i);
			GpuData* gpu_data = new GpuData();
			gpu_data->m_global = NULL;
			gpu_data->m_local = NULL;
			m_GpuData.insert(GPUDataMapPairType(i, gpu_data));
		}
	}
	return i_GPU_N;
}

void GpuBlastMultiGPUsUtils::ReleaseGPUs()
{
	if (b_useGpu)
	{
		while(!q_gpu_ids.empty())
		{
			int gpu_id = q_gpu_ids.front();
			checkCudaErrors(cudaSetDevice(gpu_id));

			GPUDataMapType::iterator itr = m_GpuData.find(gpu_id);
			if (itr != m_GpuData.end())
			{
				GpuData* gpu_data = itr->second;

				if (gpu_data != NULL)
				{
					if (gpu_data->m_global)
					{
						delete gpu_data->m_global;
					}
					if (gpu_data->m_local != NULL)
					{
						delete gpu_data->m_local;
					}
					//if (gpu_data->m_pairs != NULL)
					//{
					//	free(gpu_data->m_pairs);
					//}
					delete gpu_data;
				}
			}

			cudaDeviceReset();
			q_gpu_ids.pop();
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
		if (mt_GPU.find(p_thread_id) != mt_GPU.end())
		{
			gpu_id = mt_GPU[p_thread_id];
		}
		else
		{
 			gpu_id = q_gpu_ids.front();
			mt_GPU.insert(ThreadGPUMapPairType(p_thread_id, gpu_id));
			checkCudaErrors(cudaSetDevice(gpu_id));
			string thread_name = "GPU thread ";
			thread_name += gpu_id;
			mt_lock.SetCurrentThreadName(p_thread_id, thread_name);
			//checkCudaErrors(cudaStreamCreate(&m_GpuData[gpu_id]->stream));
			q_gpu_ids.pop();
		}
		//cout <<"Thread:"<< p_thread_id <<" GPU id:"<< gpu_id << endl;
	}
	mt_lock.SectionUnlock();
}

void GpuBlastMultiGPUsUtils::ThreadReplaceGPU()
{
	mt_lock.SectionLock();

	unsigned long p_thread_id = mt_lock.GetCurrentThreadID();
	if (p_thread_id > 0)
	{
		ThreadGPUMapType::iterator itr = mt_GPU.find(p_thread_id);
		if( itr != mt_GPU.end())
		{	
			int gpu_id = itr->second;
			q_gpu_ids.push(gpu_id);
			mt_GPU.erase(itr);
		}
	}

	mt_lock.SectionUnlock();
}
//////////////////////////////////////////////////////////////////////////

GpuData* GpuBlastMultiGPUsUtils::GetCurrentThreadGPUData()
{
	unsigned long p_thread_id = mt_lock.GetCurrentThreadID();

	//cout << "thread Id: " << p_thread_id << endl;

	if(mt_GPU.find(p_thread_id) != mt_GPU.end())
	{
		int gpu_id = mt_GPU[p_thread_id];
		checkCudaErrors(cudaSetDevice(gpu_id));
		return m_GpuData[gpu_id];
	}

	return NULL;
}

GpuBlastMultiGPUsUtils& BlastMGPUUtil = GpuBlastMultiGPUsUtils::instance();

