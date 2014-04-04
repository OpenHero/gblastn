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
		Int4 s_global_offset = scan_range_0 + s_index;
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
	Uint4 g_offset = 0;

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
		g_offset = s_index << 2;

		Int4 s_temp_2 = s[0] << 16 | s[1] << 8 | s[2];
		//s_temp = (s_temp >> top_shift) & kLutWordMask;		
		Int4 s_temp = s_temp_2 >> 2;
		checkResult(d_lookupArray, s_temp, scan_range_0, pv_array_bts, g_offset, sh_offsetpair, sh_cnt, sh_global_i, offset_pairs, total_hit);
		
		s_temp = s_temp_2 & kLutWordMask;
		checkResult(d_lookupArray, s_temp, scan_range_0, pv_array_bts, g_offset+1, sh_offsetpair, sh_cnt, sh_global_i, offset_pairs, total_hit);

		s_temp_2 = s[0] << 24 |s[1] << 16 | s[2] << 8 | s[3];
		s_temp = (s_temp_2 >> 6) & kLutWordMask;
		checkResult(d_lookupArray, s_temp, scan_range_0, pv_array_bts, g_offset+2, sh_offsetpair, sh_cnt, sh_global_i, offset_pairs, total_hit);

		s_temp = (s_temp_2 >> 4) & kLutWordMask;
		checkResult(d_lookupArray, s_temp, scan_range_0, pv_array_bts, g_offset+3, sh_offsetpair, sh_cnt, sh_global_i, offset_pairs, total_hit);

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

#define  BLOCK_SIZE 256

__global__ void gpu_blastn_scan_11_1mod4_opt(
	Uint4* subject,
	BlastOffsetPair* NCBI_RESTRICT offset_pairs, Uint4* total_hit /*max_hits*/,  
	Uint4 scan_range,
	Uint4 scan_range_0,
	Int4 top_shift,
	Int4 pv_array_bts,
	Uint4 global_size,
	PV_ARRAY_TYPE* d_lookupArray)
{
	Uint4 s_index = blockIdx.x*blockDim.x +threadIdx.x;
	Uint4 g_offset = 0;

	__shared__ Uint4 subjet_seq[BLOCK_SIZE+1];
	__shared__ BlastOffsetPair sh_offsetpair[BLOCK_SIZE*2];
	__shared__ Uint4 sh_cnt;
	__shared__ Uint4 sh_global_i;

	Int4 s_temp_2 =0;
	Int4 s_temp = 0;

	if (threadIdx.x == 0)
	{
		sh_cnt = 0;
		sh_global_i = 0;
	}
	__syncthreads();

	while(s_index < scan_range)
	{
		g_offset = s_index << 4;

		Uint4 seq1 = subject[s_index];
		Uint1* s = (Uint1*)&seq1;

		s_temp_2 = s[0] << 16 | s[1] << 8 | s[2];
		s_temp = s_temp_2 >> 2;
		checkResult(d_lookupArray, s_temp, scan_range_0, pv_array_bts, g_offset++, sh_offsetpair, sh_cnt, sh_global_i, offset_pairs, total_hit);

		s_temp = s_temp_2 & kLutWordMask;
		checkResult(d_lookupArray, s_temp, scan_range_0, pv_array_bts, g_offset++, sh_offsetpair, sh_cnt, sh_global_i, offset_pairs, total_hit);

		s_temp_2 = s[0] << 24 |s[1] << 16 | s[2] << 8 | s[3];
		s_temp = (s_temp_2 >> 6) & kLutWordMask;
		checkResult(d_lookupArray, s_temp, scan_range_0, pv_array_bts, g_offset++, sh_offsetpair, sh_cnt, sh_global_i, offset_pairs, total_hit);

		s_temp = (s_temp_2 >> 4) & kLutWordMask;
		checkResult(d_lookupArray, s_temp, scan_range_0, pv_array_bts, g_offset++, sh_offsetpair, sh_cnt, sh_global_i, offset_pairs, total_hit);


		subjet_seq[threadIdx.x] = seq1;
		if (blockIdx.x == blockDim.x)
		{
			subjet_seq[threadIdx.x+1] = subject[s_index+1];
		}
		__syncthreads();

		s = (Uint1*)(&subjet_seq[threadIdx.x]);
		s++;

		for (int i = 0; i < 3; i++)
		{
			s_temp_2 = s[0] << 16 | s[1] << 8 | s[2];
			s_temp = s_temp_2 >> 2;
			checkResult(d_lookupArray, s_temp, scan_range_0, pv_array_bts, g_offset++, sh_offsetpair, sh_cnt, sh_global_i, offset_pairs, total_hit);

			s_temp = s_temp_2 & kLutWordMask;
			checkResult(d_lookupArray, s_temp, scan_range_0, pv_array_bts, g_offset++, sh_offsetpair, sh_cnt, sh_global_i, offset_pairs, total_hit);

			s_temp_2 = s[0] << 24 |s[1] << 16 | s[2] << 8 | s[3];
			s_temp = (s_temp_2 >> 6) & kLutWordMask;
			checkResult(d_lookupArray, s_temp, scan_range_0, pv_array_bts, g_offset++, sh_offsetpair, sh_cnt, sh_global_i, offset_pairs, total_hit);

			s_temp = (s_temp_2 >> 4) & kLutWordMask;
			checkResult(d_lookupArray, s_temp, scan_range_0, pv_array_bts, g_offset++, sh_offsetpair, sh_cnt, sh_global_i, offset_pairs, total_hit);
			s++;

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

//////////////////////////////////////////////////////////////////////////
//
//////////////////////////////////////////////////////////////////////////

	__inline__ __device__ void checkResult_opt(
		PV_ARRAY_TYPE* d_lookupArray, 
		Int4 s_temp,
		Uint4 scan_range_0,
		Int4 pv_array_bts,
		Uint4 s_index,
		BlastOffsetPair* l_offsetpair,
		int &l_n)
	{
		if (s_gpu_BlastMBLookupHasHits(d_lookupArray, s_temp, pv_array_bts))
		{  
			Int4 s_global_offset = scan_range_0 + s_index;
			l_offsetpair[l_n].qs_offsets.q_off = s_temp;
			l_offsetpair[l_n++].qs_offsets.s_off = s_global_offset;
		}
	}

#define LOCAL_LEN 16
#define BLOCK_SIZE_V1 512

__global__ void gpu_blastn_scan_11_1mod4_opt_v1(
	Uint4* subject,
	BlastOffsetPair* NCBI_RESTRICT offset_pairs, Uint4* total_hit /*max_hits*/,  
	Uint4 scan_range,
	Uint4 scan_range_0,
	Int4 top_shift,
	Int4 pv_array_bts,
	Uint4 global_size,
	PV_ARRAY_TYPE* d_lookupArray)
{
	Uint4 s_index = blockIdx.x*blockDim.x +threadIdx.x;
	Uint4 g_offset = 0;

	__shared__ Uint4 subjet_seq[BLOCK_SIZE_V1+1];
	__shared__ BlastOffsetPair sh_offsetpair[2*BLOCK_SIZE_V1];
	__shared__ Uint4 sh_cnt;
	__shared__ Uint4 sh_global_i;

	BlastOffsetPair l_op[LOCAL_LEN];
	int l_n = 0;

	Int4 s_temp_2 =0;
	Int4 s_temp = 0;

	if (threadIdx.x == 0)
	{
		sh_cnt = 0;
		sh_global_i = 0;
	}
	__syncthreads();

	while(s_index < scan_range)
	{
		g_offset = s_index << 4;

		Uint4 seq1 = subject[s_index];
		Uint1* s = (Uint1*)&seq1;

		s_temp_2 = s[0] << 16 | s[1] << 8 | s[2];
		s_temp = s_temp_2 >> 2;
		checkResult_opt(d_lookupArray, s_temp, scan_range_0, pv_array_bts, g_offset++, l_op, l_n);

		s_temp = s_temp_2 & kLutWordMask;
		checkResult_opt(d_lookupArray, s_temp, scan_range_0, pv_array_bts, g_offset++, l_op, l_n);

		s_temp_2 = s[0] << 24 |s[1] << 16 | s[2] << 8 | s[3];
		s_temp = (s_temp_2 >> 6) & kLutWordMask;
		checkResult_opt(d_lookupArray, s_temp, scan_range_0, pv_array_bts, g_offset++, l_op, l_n);

		s_temp = (s_temp_2 >> 4) & kLutWordMask;
		checkResult_opt(d_lookupArray, s_temp, scan_range_0, pv_array_bts, g_offset++, l_op, l_n);


		subjet_seq[threadIdx.x] = seq1;
		if (threadIdx.x == blockDim.x-1)
		{
			subjet_seq[threadIdx.x+1] = subject[s_index+1];
		}
		__syncthreads();

		s = (Uint1*)(&subjet_seq[threadIdx.x]);
		s++;

		for (int i = 0; i < 3; i++)
		{
			s_temp_2 = s[0] << 16 | s[1] << 8 | s[2];
			s_temp = s_temp_2 >> 2;
			checkResult_opt(d_lookupArray, s_temp, scan_range_0, pv_array_bts, g_offset++, l_op, l_n);

			s_temp = s_temp_2 & kLutWordMask;
			checkResult_opt(d_lookupArray, s_temp, scan_range_0, pv_array_bts, g_offset++, l_op, l_n);

			s_temp_2 = s[0] << 24 |s[1] << 16 | s[2] << 8 | s[3];
			s_temp = (s_temp_2 >> 6) & kLutWordMask;
			checkResult_opt(d_lookupArray, s_temp, scan_range_0, pv_array_bts, g_offset++, l_op, l_n);

			s_temp = (s_temp_2 >> 4) & kLutWordMask;
			checkResult_opt(d_lookupArray, s_temp, scan_range_0, pv_array_bts, g_offset++, l_op, l_n);
			s++;

		}

		for (int i = 0; i < LOCAL_LEN; i++)
		{
			if (i < l_n)
			{
				Uint4 index_offset = atomicAdd(&sh_cnt, 1);
				sh_offsetpair[index_offset] = l_op[i];
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

/////////////////////////////////////////////////////////////////////////////////////
/*
   const Int4 kTemplateLength = 18;
   Uint4 lo = 0; 
   Uint4 hi = 0;



   index = scan_range[0] - (scan_range[0] % COMPRESSION_RATIO);
   while(index < scan_range[0] + kTemplateLength) {
      hi = (hi << 8) | (lo >> 24);
      lo = lo << 8 | *s++;
      index += COMPRESSION_RATIO;
   }

   switch (index - (scan_range[0] + kTemplateLength)) {
   case 1: 
       goto base_3;
   case 2: 
       goto base_2;
   case 3: 
       s--;
       lo = (lo >> 8) | (hi << 24);
       hi = hi >> 8;
       goto base_1;
   }

   while (scan_range[0] <= scan_range[1]) {

      index = ((lo & 0x00000003)      ) |
              ((lo & 0x000000f0) >>  2) |
              ((lo & 0x00003c00) >>  4) |
              ((lo & 0x00030000) >>  6) |
              ((lo & 0x03c00000) >> 10) |
              ((lo & 0xf0000000) >> 12) |
              ((hi & 0x0000000c) << 18);
      MB_ACCESS_HITS();
      scan_range[0]++;

base_1:
      if (scan_range[0] > scan_range[1])
         break;

      hi = (hi << 8) | (lo >> 24);
      lo = lo << 8 | *s++;

      index = ((lo & 0x000000c0) >>  6) |
              ((lo & 0x00003c00) >>  8) |
              ((lo & 0x000f0000) >> 10) |
              ((lo & 0x00c00000) >> 12) |
              ((lo & 0xf0000000) >> 16) |
              ((hi & 0x0000003c) << 14) |
              ((hi & 0x00000300) << 12);
      MB_ACCESS_HITS();
      scan_range[0]++;

base_2:
      if (scan_range[0] > scan_range[1])
         break;

      index = ((lo & 0x00000030) >>  4) |
              ((lo & 0x00000f00) >>  6) |
              ((lo & 0x0003c000) >>  8) |
              ((lo & 0x00300000) >> 10) |
              ((lo & 0x3c000000) >> 14) |
              ((hi & 0x0000000f) << 16) |
              ((hi & 0x000000c0) << 14);
      MB_ACCESS_HITS();
      scan_range[0]++;

base_3:
      if (scan_range[0] > scan_range[1])
         break;

      index = ((lo & 0x0000000c) >>  2) |
              ((lo & 0x000003c0) >>  4) |
              ((lo & 0x0000f000) >>  6) |
              ((lo & 0x000c0000) >>  8) |
              ((lo & 0x0f000000) >> 12) |
              ((lo & 0xc0000000) >> 14) |
              ((hi & 0x00000003) << 18) |
              ((hi & 0x00000030) << 16);
      MB_ACCESS_HITS();
      scan_range[0]++;
   }
   return total_hits;
*/
__global__ void s_gpu_MB_DiscWordScanSubject_11_18_1(
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
	Uint4 g_offset = 0;

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
		Uint4 lo = 0; 
		Uint4 hi = 0;

		Uint1* s = subject+ (s_index<<2);
		g_offset = s_index << 2;

		for(int i = 0; i < 5; i++)
		{
			hi = (hi << 8) | (lo >> 24);
			lo = lo << 8 | *s++;
		}
		Int4 s_temp = ((lo & 0x00000030) >>  4) |
              ((lo & 0x00000f00) >>  6) |
              ((lo & 0x0003c000) >>  8) |
              ((lo & 0x00300000) >> 10) |
              ((lo & 0x3c000000) >> 14) |
              ((hi & 0x0000000f) << 16) |
              ((hi & 0x000000c0) << 14);		
		checkResult(d_lookupArray, s_temp, scan_range_0, pv_array_bts, g_offset, sh_offsetpair, sh_cnt, sh_global_i, offset_pairs, total_hit);
		
		s_temp = ((lo & 0x0000000c) >>  2) |
              ((lo & 0x000003c0) >>  4) |
              ((lo & 0x0000f000) >>  6) |
              ((lo & 0x000c0000) >>  8) |
              ((lo & 0x0f000000) >> 12) |
              ((lo & 0xc0000000) >> 14) |
              ((hi & 0x00000003) << 18) |
              ((hi & 0x00000030) << 16);
		checkResult(d_lookupArray, s_temp, scan_range_0, pv_array_bts, g_offset+1, sh_offsetpair, sh_cnt, sh_global_i, offset_pairs, total_hit);

		s_temp = ((lo & 0x00000003)      ) |
              ((lo & 0x000000f0) >>  2) |
              ((lo & 0x00003c00) >>  4) |
              ((lo & 0x00030000) >>  6) |
              ((lo & 0x03c00000) >> 10) |
              ((lo & 0xf0000000) >> 12) |
              ((hi & 0x0000000c) << 18);
		checkResult(d_lookupArray, s_temp, scan_range_0, pv_array_bts, g_offset+2, sh_offsetpair, sh_cnt, sh_global_i, offset_pairs, total_hit);

		hi = (hi << 8) | (lo >> 24);
		lo = lo << 8 | *s++;

		s_temp = ((lo & 0x000000c0) >>  6) |
			((lo & 0x00003c00) >>  8) |
			((lo & 0x000f0000) >> 10) |
			((lo & 0x00c00000) >> 12) |
			((lo & 0xf0000000) >> 16) |
			((hi & 0x0000003c) << 14) |
			((hi & 0x00000300) << 12);
		checkResult(d_lookupArray, s_temp, scan_range_0, pv_array_bts, g_offset+3, sh_offsetpair, sh_cnt, sh_global_i, offset_pairs, total_hit);

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
