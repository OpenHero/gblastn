#ifndef __GPU_BLASTN_NA_SCAN_KERNEL_V3_H__
#define __GPU_BLASTN_NA_SCAN_KERNEL_V3_H__

#include <algo/blast/core/blast_nalookup.h>
#include <algo/blast/core/blast_nascan.h>
#include <algo/blast/core/blast_util.h>

#define  USE_TEXTURE 1

#if(USE_TEXTURE)
texture<PV_ARRAY_TYPE,1,cudaReadModeElementType> tx_pv_array;
#define   LOAD_PV(i) tex1Dfetch(tx_pv_array, i)
#define  SET_PVARRAY_BASE checkCudaErrors( cudaBindTexture(0, tx_pv_array, d_lookupArray) )
#else
#define  LOAD_PV(i) d_lookupArray[i]
#define SET_PVARRAY_BASE
#endif

#define SHARE_MEM_SIZE 1024
/**
* Determine if this subject word occurs in the query.
* @param lookup The lookup table to read from. [in]
* @param index The index value of the word to retrieve. [in]
* @return 1 if there are hits, 0 otherwise.
*/
__device__ __inline__ int s_gpu_BlastMBLookupHasHits(PV_ARRAY_TYPE* d_lookupArray, Uint4 index, Uint4 pv_array_bts)
{
	Uint4 array_index = index >> pv_array_bts;

	if( (LOAD_PV(array_index)) & ((PV_ARRAY_TYPE)1 << ((index) & PV_ARRAY_MASK)) )
		return 1;
	else
		return 0;
}

__global__ void gpu_blastn_scan_Any_v3(
	const Uint4* subject,  
	BlastOffsetPair* NCBI_RESTRICT offset_pairs, Uint4* total_hit /*max_hits*/,  
	Uint4 scan_range,
	Uint4 scan_range_0,
	Int4 mask, // = mb_lt->hashsize - 1;,
	Int4 shift,
	Uint4 pv_array_bts,
	Uint4 global_size,
	PV_ARRAY_TYPE* d_lookupArray)
{
	Uint4 s_index = blockIdx.x*blockDim.x +threadIdx.x;
 	
	__shared__ Uint4 sh_cnt;
	__shared__ Uint4 sh_global_i;
	__shared__ BlastOffsetPair sh_offsetpair[SHARE_MEM_SIZE];


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

		Uint4 s_temp = s[0] << 24 | s[1] << 16 | s[2] << 8 | s[3];
		s_temp = ((s_temp >> shift) & mask);


		if (s_gpu_BlastMBLookupHasHits(d_lookupArray, s_temp, pv_array_bts))
		{  
			Uint4 s_global_offset = scan_range_0 + s_index << 4;
			Uint4 index_offset = atomicAdd(&sh_cnt, 1);
			sh_offsetpair[index_offset].qs_offsets.s_off = s_global_offset;
			sh_offsetpair[index_offset].qs_offsets.q_off = s_temp;
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
		}
		__syncthreads();
		s_index += global_size;
	}

	__syncthreads();

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


__constant__ Int4 kLutWordMask = (1 << (2 * 11)) - 1;

__global__ void gpu_blastn_scan_11_2mod4_v3(
	const Uint4* subject,
	BlastOffsetPair* NCBI_RESTRICT offset_pairs, Uint4* total_hit /*max_hits*/,  
	Uint4 scan_range,
	Uint4 scan_range_0,
	Int4 top_shift,
	Int4 pv_array_bts,
	Uint4 global_size,
	PV_ARRAY_TYPE* d_lookupArray)
{
	Uint4 s_index = blockIdx.x*blockDim.x +threadIdx.x;

	__shared__ BlastOffsetPair sh_offsetpair[SHARE_MEM_SIZE];
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

		Int4 s_temp = s[0] << 16 | s[1] << 8 | s[2];
		s_temp = (s_temp >> top_shift) & kLutWordMask;		

		if (s_gpu_BlastMBLookupHasHits(d_lookupArray, s_temp, pv_array_bts))
		{  
			Int4 s_global_offset = scan_range_0 + s_index << 4;
			Uint4 index_offset = atomicAdd(&sh_cnt, 1);
			sh_offsetpair[index_offset].qs_offsets.q_off = s_temp;
			sh_offsetpair[index_offset].qs_offsets.s_off = s_global_offset;
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

	__syncthreads();
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


__inline__ __device__ void checkResult(PV_ARRAY_TYPE* d_lookupArray, 
	Int4 s_temp,
	Uint4 scan_range_0,
	Int4 pv_array_bts,
	Uint4 s_index,
	BlastOffsetPair* sh_offsetpair,
	Uint4& sh_cnt,
	Uint4& sh_global_i,
	BlastOffsetPair* NCBI_RESTRICT offset_pairs,
	Uint4* total_hit)
{
	if (s_gpu_BlastMBLookupHasHits(d_lookupArray, s_temp, pv_array_bts))
	{  
		Int4 s_global_offset = scan_range_0 + s_index << 4;
		Uint4 index_offset = atomicAdd(&sh_cnt, 1);
		sh_offsetpair[index_offset].qs_offsets.q_off = s_temp;
		sh_offsetpair[index_offset].qs_offsets.s_off = s_global_offset;
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
}

__global__ void gpu_blastn_scan_11_1mod4_v3(
	Uint1* subject,
	BlastOffsetPair* NCBI_RESTRICT offset_pairs, Uint4* total_hit /*max_hits*/,  
	Uint4 scan_range,
	Uint4 scan_range_0,
	Int4 top_shift,
	Int4 pv_array_bts,
	Uint4 global_size,
	PV_ARRAY_TYPE* d_lookupArray)
{
	Uint4 s_index = blockIdx.x*blockDim.x +threadIdx.x;

	__shared__ BlastOffsetPair sh_offsetpair[SHARE_MEM_SIZE];
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
		//Uint4  s_temp_1 = subject[s_index];
		//Uint1* s = (Uint1*)&s_temp_1;
		Uint1* s = subject +s_index;

		Int4 s_temp_2 = s[0] << 16 | s[1] << 8 | s[2];
		//s_temp = (s_temp >> top_shift) & kLutWordMask;		
		Int4 s_temp = s_temp_2 >> 2;
		checkResult(d_lookupArray, s_temp, scan_range_0, pv_array_bts, s_index, sh_offsetpair, sh_cnt, sh_global_i, offset_pairs, total_hit);
		
		s_temp = s_temp_2 & kLutWordMask;
		checkResult(d_lookupArray, s_temp, scan_range_0, pv_array_bts, s_index, sh_offsetpair, sh_cnt, sh_global_i, offset_pairs, total_hit);

		s_temp_2 = s[0] << 24 |s[1] << 16 | s[2] << 8 | s[3];
		s_temp = (s_temp_2 >> 6) & kLutWordMask;
		checkResult(d_lookupArray, s_temp, scan_range_0, pv_array_bts, s_index, sh_offsetpair, sh_cnt, sh_global_i, offset_pairs, total_hit);

		s_temp = (s_temp_2 >> 2) & kLutWordMask;
		checkResult(d_lookupArray, s_temp, scan_range_0, pv_array_bts, s_index, sh_offsetpair, sh_cnt, sh_global_i, offset_pairs, total_hit);

		s_index += global_size;
	}

	__syncthreads();
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

#endif 
