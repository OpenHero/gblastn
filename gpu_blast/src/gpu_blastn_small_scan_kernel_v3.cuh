#ifndef __GPU_BLASTN_SMALL_SCAN_KERNEL_V3_H__
#define __GPU_BLASTN_SMALL_SCAN_KERNEL_V3_H__

#define SHARE_MEM_SMALL_SIZE 512

#define USE_TEXTURE 1

#if(USE_TEXTURE)
//texture<float, 1, cudaReadModeElementType> texFloat;
texture<Int2,1,cudaReadModeElementType> tx_backbone;
#define   LOAD_INT2(i) tex1Dfetch(tx_backbone, i)
#define  SET_INT2_BASE checkCudaErrors( cudaBindTexture(0, tx_backbone, d_backbone) )
#else
#define  LOAD_INT2(i) d_backbone[i]
#define SET_INT2_BASE
#endif

__global__ void gpu_blastn_scan_8_1mod4_v3(const Uint4* subject,
	BlastOffsetPair* NCBI_RESTRICT offset_pairs, 
	BlastOffsetPair* NCBI_RESTRICT over_offset_pairs,
	Uint4* total_hit /*max_hits*/,
	Uint4* over_total_hit,
	Uint4 scan_range,
	Uint4 scan_range_0,
	Uint4 global_size,
	Int2* d_backbone)
{
	Uint4 s_index = blockIdx.x*blockDim.x +threadIdx.x;

	__shared__ BlastOffsetPair sh_offsetpair[SHARE_MEM_SMALL_SIZE];
	__shared__ Uint4 sh_cnt;
	__shared__ Uint4 sh_global_i;

	if (threadIdx.x == 0)
	{
		sh_cnt = 0;
		sh_global_i = 0;
	}
	__syncthreads();

	while(s_index < scan_range)
	{
		Uint4  s_temp_1 = subject[s_index];
		Uint1* s = (Uint1*)&s_temp_1;

		Uint4 s_temp = s[0] << 8 | s[1];		
		Int2 s_temp_value = LOAD_INT2(s_temp);		

		Uint4 s_global_offset = scan_range_0 + s_index << 4;
		if (s_temp_value > -1)
		{  
			Uint4 index_offset = atomicAdd(&sh_cnt, 1);
			sh_offsetpair[index_offset].qs_offsets.q_off = s_temp_value;
			sh_offsetpair[index_offset].qs_offsets.s_off = s_global_offset;
		}
		if(s_temp_value < -1)
		{
			Uint4 index_offset = atomicAdd(over_total_hit, 1);
			over_offset_pairs[index_offset].qs_offsets.q_off = -s_temp_value;
			over_offset_pairs[index_offset].qs_offsets.s_off = s_global_offset;
		}

		__syncthreads();

		if (sh_cnt >= blockDim.x)
		{  
			int offset_id = threadIdx.x;

			if (offset_id == 0)
			{
				sh_global_i = atomicAdd(total_hit, sh_cnt);

			}
			__syncthreads();			

			while (offset_id < sh_cnt)
			{ 
				offset_pairs[sh_global_i + offset_id] = sh_offsetpair[offset_id];
				offset_id += blockDim.x;
			}
			if (threadIdx.x == 0)
			{
				sh_cnt = 0; 
			} 
			__syncthreads();
		}

		s_index += global_size;
	}

	if (sh_cnt > 0)
	{
		int offset_id = threadIdx.x;

		if (offset_id == 0)
		{
			sh_global_i = atomicAdd(total_hit, sh_cnt);

		}
		__syncthreads();
		if (offset_id < sh_cnt)
		{
			offset_pairs[sh_global_i + offset_id] = sh_offsetpair[offset_id];
		}
	}
}

//const Int4 kLutWordLength = 8;
#define kLutWordLength_8_4 8
__constant__ Int4 kLutWordMask_8_4 = (1 << (2 * kLutWordLength_8_4)) - 1;

__global__ void gpu_blastn_scan_8_4(const Uint4* subject,
	BlastOffsetPair* NCBI_RESTRICT offset_pairs, 
	BlastOffsetPair* NCBI_RESTRICT over_offset_pairs,
	Uint4* total_hit /*max_hits*/,
	Uint4* over_total_hit,
	Uint4 scan_range,
	Uint4 scan_range_0,
	Uint4 global_size,
	Int2* d_backbone)
{
	Uint4 s_index = blockIdx.x*blockDim.x +threadIdx.x;

	__shared__ BlastOffsetPair sh_offsetpair[SHARE_MEM_SMALL_SIZE];
	__shared__ Uint4 sh_cnt;
	__shared__ Uint4 sh_global_i;

	if (threadIdx.x == 0)
	{
		sh_cnt = 0;
		sh_global_i = 0;
	}
	__syncthreads();

	Int4 init_index;
	Int4 index;

	while(s_index < scan_range)
	{
		Uint4  s_temp_1 = subject[s_index];
		Uint1* s = (Uint1*)&s_temp_1;

		init_index = s[0];

		Uint4 index = init_index << 8 | s[1];
		index = init_index & kLutWordMask_8_4;		
		Int2 s_temp_value = LOAD_INT2(index);		

		Uint4 s_global_offset = scan_range_0 + s_index << 4;
		if (s_temp_value > -1)
		{  
			Uint4 index_offset = atomicAdd(&sh_cnt, 1);
			sh_offsetpair[index_offset].qs_offsets.q_off = s_temp_value;
			sh_offsetpair[index_offset].qs_offsets.s_off = s_global_offset;
		}
		if(s_temp_value < -1)
		{
			Uint4 index_offset = atomicAdd(over_total_hit, 1);
			over_offset_pairs[index_offset].qs_offsets.q_off = -s_temp_value;
			over_offset_pairs[index_offset].qs_offsets.s_off = s_global_offset;
		}

		__syncthreads();

		if (sh_cnt >= blockDim.x)
		{  
			int offset_id = threadIdx.x;

			if (offset_id == 0)
			{
				sh_global_i = atomicAdd(total_hit, sh_cnt);

			}
			__syncthreads();			

			while (offset_id < sh_cnt)
			{ 
				offset_pairs[sh_global_i + offset_id] = sh_offsetpair[offset_id];
				offset_id += blockDim.x;
			}
			if (threadIdx.x == 0)
			{
				sh_cnt = 0; 
			} 
			__syncthreads();
		}

		s_index += global_size;
	}

	if (sh_cnt > 0)
	{
		int offset_id = threadIdx.x;

		if (offset_id == 0)
		{
			sh_global_i = atomicAdd(total_hit, sh_cnt);

		}
		__syncthreads();
		if (offset_id < sh_cnt)
		{
			offset_pairs[sh_global_i + offset_id] = sh_offsetpair[offset_id];
		}
	}
}


#endif //__GPU_BLASTN_SMALL_SCAN_KERNEL_H__