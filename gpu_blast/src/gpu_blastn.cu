#pragma warning (disable:4819)

#include <stdio.h>
#include <stdlib.h>
#include <Windows.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>
//#include <cutil.h>

#include <algo/blast/core/blast_nalookup.h>
#include <algo/blast/core/blast_nascan.h>
#include <algo/blast/core/blast_util.h>
#include <algo/blast/core/lookup_wrap.h>         //for MAX_DBSEQ_LEN
#include <algo/blast/core/blast_gapalign.h>      //for OFFSET_ARRAY_SIZE  

#include <algo/blast/gpu_blast/gpu_blastn_GPU_setting.h>


#include "gpu_blastn_ungapped_extension_functions.h"
#include "gpu_blastn_scan_kernel.cuh"
#include "gpu_blastn_mini_extension_kernel.cuh"
#include "gpu_blastn_lookup_hash_kernel.cuh"

#include <algo/blast/gpu_blast/gpu_logfile.h>

#if USE_OPENMP
#include <omp.h>
#endif

/** Wrapper structure for different types of cuda structures */
typedef struct cudaAuxWrap {
	unsigned int *exact_hits_num;
	Int4 * hashtable;
	Int4 * next_pos;
	Uint1 * query;
	Uint1 * subject;
	BlastOffsetPair * exact_offset_pairs;
} cudaAuxWrap;


typedef struct cudaScanAuxWrap 
{
	//cudaArray* lookupArray;
	PV_ARRAY_TYPE* lookupArray;
	BlastOffsetPair* offsetPairs;
	Uint4* subject;
	Uint4* total_hits;
} cudaScanAuxWrap;

bool is_initGPUMem = false;

cudaAuxWrap cuda_wrap;

cudaScanAuxWrap cuda_scanAuxWrap;

void s_initLookupTableTexture(Int4 size, Uint4* lookuptable)
{
	//cudaArray* d_lookupArray = NULL;
	//create cuda array
	//cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
	//checkCudaErrors(cudaMallocArray( &d_lookupArray, &channelDesc, size, 1 )); 
	//checkCudaErrors(cudaMemcpyToArray( d_lookupArray, 0, 0, lookuptable, size * sizeof (float), cudaMemcpyHostToDevice));

	// Bind the array to the texture
	//checkCudaErrors(cudaBindTextureToArray( tx_pv_array, d_lookupArray, channelDesc));
	PV_ARRAY_TYPE* d_lookupArray = NULL;
	checkCudaErrors(cudaMalloc((void**)&d_lookupArray,  size * sizeof(PV_ARRAY_TYPE)));
	checkCudaErrors(cudaMemcpy(d_lookupArray, lookuptable, size * sizeof(PV_ARRAY_TYPE), cudaMemcpyHostToDevice));

	SET_PVARRAY_BASE;

	cuda_scanAuxWrap.lookupArray = d_lookupArray;
}

void s_freeLookupTableTexture()
{
	checkCudaErrors(cudaUnbindTexture(tx_pv_array));
	//checkCudaErrors(cudaFreeArray(cuda_scanAuxWrap.lookupArray));
	checkCudaErrors(cudaFree(cuda_scanAuxWrap.lookupArray));
}

void s_initScanGPUMemory(LookupTableWrap* lookup_wrap)
{
	BlastOffsetPair* d_offsetPairs = NULL;
	Uint4* d_subject = NULL;
	Uint4* d_total_hits = NULL;

	checkCudaErrors(cudaMalloc((void **)&d_total_hits,sizeof(Uint4)));
	checkCudaErrors(cudaMalloc((void **)&d_subject, MAX_DBSEQ_LEN*sizeof(Uint4)/4));
	checkCudaErrors(cudaMalloc((void **)&d_offsetPairs, OFFSET_ARRAY_SIZE * sizeof(BlastOffsetPair)));

	cuda_scanAuxWrap.offsetPairs = d_offsetPairs;
	cuda_scanAuxWrap.subject = d_subject;
	cuda_scanAuxWrap.total_hits = d_total_hits;


	BlastMBLookupTable* mb_lt = (BlastMBLookupTable*) lookup_wrap->lut;
	Int4 pv_size = mb_lt->hashsize/(1<<mb_lt->pv_array_bts);
	s_initLookupTableTexture(pv_size, mb_lt->pv_array);
}

void s_freeScanGPUMemory()
{
	cudaFree(cuda_scanAuxWrap.total_hits);
	cudaFree(cuda_scanAuxWrap.subject);
	cudaFree(cuda_scanAuxWrap.offsetPairs);
	s_freeLookupTableTexture();
}

/** Scan the compressed subject sequence, returning 11-letter word hits
* with stride two plus a multiple of four. Assumes a megablast lookup table
* @param lookup_wrap Pointer to the (wrapper to) lookup table [in]
* @param subject The (compressed) sequence to be scanned for words [in]
* @param offset_pairs Array of query and subject positions where words are 
*                found [out]
* @param max_hits The allocated size of the above array - how many offsets 
*        can be returned [in]
* @param scan_range The starting and ending pos to be scanned [in] 
*        on exit, scan_range[0] is updated to be the stopping pos [out]
*/
extern "C" Int4 
	s_gpu_MBScanSubject_11_2Mod4(const LookupTableWrap* lookup_wrap,
	const BLAST_SequenceBlk* subject,
	BlastOffsetPair* NCBI_RESTRICT offset_pairs, Int4 max_hits,  
	Int4* scan_range)
{
	BlastMBLookupTable* mb_lt = (BlastMBLookupTable*) lookup_wrap->lut;
	Uint4 total_hits = 0;
	Int4 top_shift; 

	max_hits -= mb_lt->longest_chain;
	ASSERT(lookup_wrap->lut_type == eMBLookupTable);
	ASSERT(mb_lt->lut_word_length == 11);
	if(scan_range[0] > scan_range[1]) return 0;

	top_shift = 2;


	Int4 pv_array_bts = mb_lt->pv_array_bts;
	Uint4 scan_range_temp = (scan_range[1]+mb_lt->lut_word_length - scan_range[0]);

	slogfile.Start();
	checkCudaErrors(cudaMemcpy(cuda_scanAuxWrap.subject, subject->sequence, (scan_range_temp) /4, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemset(cuda_scanAuxWrap.total_hits, 0, sizeof(unsigned int)));  //初始化为0
	
	slogfile.End();
	slogfile.addTotalTime("Scan CPU -> GPU Memory Time",slogfile.elaplsedTime(), false);

	dim3 blockSize(512);
	dim3 gridSize;
	scan_range_temp = (scan_range_temp/16) + (scan_range_temp%16);
	gridSize.x = (scan_range_temp+ blockSize.x -1)/ blockSize.x;
	
	slogfile.KernelStart();
	gpu_blastn_scan_11_2mod4<<< gridSize, blockSize >>>(cuda_scanAuxWrap.subject, 
		cuda_scanAuxWrap.offsetPairs, 
		cuda_scanAuxWrap.total_hits, 
		scan_range_temp, 
		scan_range[0], 
		top_shift, 
		pv_array_bts,
		cuda_scanAuxWrap.lookupArray);
	getLastCudaError("gpu_blastn_scan_11_2mod4() execution failed.\n");
	
	slogfile.KernelEnd();
	slogfile.addTotalTime("Scan Kernel Time", slogfile.KernelElaplsedTime(), false);

	checkCudaErrors(cudaMemcpy(&total_hits, cuda_scanAuxWrap.total_hits, sizeof(Uint4), cudaMemcpyDeviceToHost));

#if !GPU_EXT_RUN
	checkCudaErrors(cudaMemcpy(offset_pairs, cuda_scanAuxWrap.offsetPairs, total_hits * sizeof(BlastOffsetPair), cudaMemcpyDeviceToHost));
#endif

	scan_range[0] = scan_range[1]+11;

	return total_hits;
}


extern "C" Int4 
	s_gpu_MBScanSubject_11_2Mod4_v1(const LookupTableWrap* lookup_wrap,
	const BLAST_SequenceBlk* subject,
	BlastOffsetPair* NCBI_RESTRICT offset_pairs, Int4 max_hits,  
	Int4* scan_range)
{
	BlastMBLookupTable* mb_lt = (BlastMBLookupTable*) lookup_wrap->lut;
	Uint4 total_hits = 0;
	Int4 top_shift; 

	max_hits -= mb_lt->longest_chain;
	ASSERT(lookup_wrap->lut_type == eMBLookupTable);
	ASSERT(mb_lt->lut_word_length == 11);
	if(scan_range[0] > scan_range[1]) return 0;

	top_shift = 2;
	Int4 pv_array_bts = mb_lt->pv_array_bts;
	Uint4 scan_range_temp = (scan_range[1]+mb_lt->lut_word_length - scan_range[0]);

	slogfile.Start();
	checkCudaErrors(cudaMemcpy(cuda_scanAuxWrap.subject, subject->sequence, (scan_range_temp) /4, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemset(cuda_scanAuxWrap.total_hits, 0, sizeof(unsigned int)));  //初始化为0

	slogfile.End();
	slogfile.addTotalTime("Scan CPU -> GPU Memory Time",slogfile.elaplsedTime(), false);

	dim3 blockSize(512);
	dim3 gridSize;
	scan_range_temp = (scan_range_temp/16) + (scan_range_temp%16);
	gridSize.x = (scan_range_temp+ blockSize.x -1)/ blockSize.x;

	//printf("gpu_blastn_scan_11_2mod4:%d \n", gridSize.x);

	slogfile.KernelStart();

	gpu_blastn_scan_11_2mod4<<< gridSize, blockSize >>>(cuda_scanAuxWrap.subject, 
		cuda_scanAuxWrap.offsetPairs, 
		cuda_scanAuxWrap.total_hits, 
		scan_range_temp, 
		scan_range[0], 
		top_shift, 
		pv_array_bts,
		cuda_scanAuxWrap.lookupArray);

	getLastCudaError("gpu_blastn_scan_11_2mod4() execution failed.\n");

	slogfile.KernelEnd();
	slogfile.addTotalTime("Scan Kernel Time", slogfile.KernelElaplsedTime(), false);

	checkCudaErrors(cudaMemcpy(&total_hits, cuda_scanAuxWrap.total_hits, sizeof(Uint4), cudaMemcpyDeviceToHost));


	Int4 threadNum = 512;
	Int4 blockNum = (total_hits + threadNum - 1)/threadNum;
	dim3 gridDim(blockNum, 1);
	dim3 blockDim(threadNum, 1);

	//printf("kernel_lookupInBigHashTable:%d \n", blockNum);

	//checkCudaErrors(cudaMemcpy(offset_pairs, d_offsetPairs, total_hits * sizeof(BlastOffsetPair), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemset(cuda_wrap.exact_hits_num, 0, sizeof(unsigned int)));  //初始化为0

	slogfile.KernelStart();

	kernel_lookupInBigHashTable<<<gridDim,blockDim>>>(cuda_wrap.hashtable,
		cuda_wrap.next_pos,
		total_hits,
		cuda_scanAuxWrap.offsetPairs,
		cuda_wrap.exact_offset_pairs,
		cuda_wrap.exact_hits_num
		);
	getLastCudaError("kernel_lookupInBigHashTable() execution failed.\n");
	checkCudaErrors(cudaMemcpy(&total_hits, cuda_wrap.exact_hits_num, sizeof(unsigned int), cudaMemcpyDeviceToHost));
	slogfile.KernelEnd();
	slogfile.addTotalTime("LookUpTableHash Time v1", slogfile.KernelElaplsedTime(), false);
	slogfile.addTotalNum("Kernel_lookupInBigHashTable hits", total_hits, false);



#if !GPU_EXT_RUN
	checkCudaErrors(cudaMemcpy(offset_pairs, cuda_wrap.exact_offset_pairs, total_hits * sizeof(BlastOffsetPair), cudaMemcpyDeviceToHost));
#endif
	//slogfile.m_file << "Scan hits:"<<total_hits<<endl;

	scan_range[0] = scan_range[1]+11;

	return total_hits;
}

extern "C" Int4 
	s_gpu_MBScanSubject_11_2Mod4_v1_scankernel_v1(const LookupTableWrap* lookup_wrap,
	const BLAST_SequenceBlk* subject,
	BlastOffsetPair* NCBI_RESTRICT offset_pairs, Int4 max_hits,  
	Int4* scan_range)
{
	BlastMBLookupTable* mb_lt = (BlastMBLookupTable*) lookup_wrap->lut;
	Uint4 total_hits = 0;
	Int4 top_shift; 

	max_hits -= mb_lt->longest_chain;
	ASSERT(lookup_wrap->lut_type == eMBLookupTable);
	ASSERT(mb_lt->lut_word_length == 11||
		mb_lt->lut_word_length == 12);
	if(scan_range[0] > scan_range[1]) return 0;

	top_shift = 2;
	Int4 pv_array_bts = mb_lt->pv_array_bts;
	Uint4 scan_range_temp = (scan_range[1]+mb_lt->lut_word_length - scan_range[0]);

	slogfile.Start();
	checkCudaErrors(cudaMemcpy(cuda_scanAuxWrap.subject, subject->sequence, (scan_range_temp)/4, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemset(cuda_scanAuxWrap.total_hits, 0, sizeof(unsigned int)));  //初始化为0

	slogfile.End();
	slogfile.addTotalTime("Scan CPU -> GPU Memory Time",slogfile.elaplsedTime(), false);

	static int blocksize_x = 128;
	dim3 blockSize(blocksize_x);
	dim3 gridSize;
	scan_range_temp = (scan_range_temp/16) + (scan_range_temp%16);
	gridSize.x = (scan_range_temp+ blockSize.x -1)/ blockSize.x;

	static int max_grid_size = 4096;
	if (gridSize.x > max_grid_size)
	{
		gridSize.x = max_grid_size;
	}

	Uint4 global_size = gridSize.x;
	global_size *= blocksize_x;

	slogfile.KernelStart();

	gpu_blastn_scan_11_2mod4_v1<<< gridSize, blockSize >>>(
		cuda_scanAuxWrap.subject, 
		cuda_scanAuxWrap.offsetPairs, 
		cuda_scanAuxWrap.total_hits, 
		scan_range_temp, 
		scan_range[0], 
		top_shift, 
		pv_array_bts,
		global_size,
		cuda_scanAuxWrap.lookupArray);

	
	getLastCudaError("gpu_blastn_scan_11_2mod4() execution failed.\n");

	slogfile.KernelEnd();
	slogfile.addTotalTime("Scan Kernel Time", slogfile.KernelElaplsedTime(), false);

	checkCudaErrors(cudaMemcpy(&total_hits, cuda_scanAuxWrap.total_hits, sizeof(Uint4), cudaMemcpyDeviceToHost));

	Int4 threadNum = 512;
	Int4 blockNum = (total_hits + threadNum - 1)/threadNum;
	dim3 gridDim(blockNum, 1);
	dim3 blockDim(threadNum, 1);

	//checkCudaErrors(cudaMemcpy(offset_pairs, d_offsetPairs, total_hits * sizeof(BlastOffsetPair), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemset(cuda_wrap.exact_hits_num, 0, sizeof(unsigned int)));  //初始化为0

	slogfile.KernelStart();

	kernel_lookupInBigHashTable<<<gridDim,blockDim>>>(
		cuda_wrap.hashtable,
		cuda_wrap.next_pos,
		total_hits,
		cuda_scanAuxWrap.offsetPairs,
		cuda_wrap.exact_offset_pairs,
		cuda_wrap.exact_hits_num
		);
	getLastCudaError("kernel_lookupInBigHashTable() execution failed.\n");
	checkCudaErrors(cudaMemcpy(&total_hits, cuda_wrap.exact_hits_num, sizeof(unsigned int), cudaMemcpyDeviceToHost));
	slogfile.KernelEnd();
	slogfile.addTotalTime("LookUpTableHash Time v1", slogfile.KernelElaplsedTime(), false);
	slogfile.addTotalNum("Kernel_lookupInBigHashTable hits", total_hits);

#if !GPU_EXT_RUN
	checkCudaErrors(cudaMemcpy(offset_pairs, cuda_wrap.exact_offset_pairs, total_hits * sizeof(BlastOffsetPair), cudaMemcpyDeviceToHost));
#endif
	//slogfile.m_file << "Scan hits:"<<total_hits<<endl;

	scan_range[0] = scan_range[1]+11;

	return total_hits;
}


extern "C" Int4 
	s_gpu_MBScanSubject_Any_scankernel_v1(const LookupTableWrap* lookup_wrap,
	const BLAST_SequenceBlk* subject,
	BlastOffsetPair* NCBI_RESTRICT offset_pairs, Int4 max_hits,  
	Int4* scan_range)
{
	BlastMBLookupTable* mb_lt = (BlastMBLookupTable*) lookup_wrap->lut;
	Uint4 total_hits = 0;
	Int4 shift; 
	Int4 mask = mb_lt->hashsize - 1;
	max_hits -= mb_lt->longest_chain;
	ASSERT(lookup_wrap->lut_type == eMBLookupTable);
	ASSERT(mb_lt->lut_word_length == 11||
		mb_lt->lut_word_length == 12);
	if(scan_range[0] > scan_range[1]) return 0;

	shift = 2*(16 - (scan_range[0] % COMPRESSION_RATIO + mb_lt->lut_word_length));;
	Int4 pv_array_bts = mb_lt->pv_array_bts;
	Uint4 scan_range_temp = (scan_range[1]+11 - scan_range[0]);

	slogfile.Start();
	checkCudaErrors(cudaMemcpy(cuda_scanAuxWrap.subject, subject->sequence, (scan_range_temp) / 4, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemset(cuda_scanAuxWrap.total_hits, 0, sizeof(unsigned int)));  //初始化为0

	slogfile.End();
	slogfile.addTotalTime("Scan CPU -> GPU Memory Time",slogfile.elaplsedTime(), false);

	static int blocksize_x = 128;
	dim3 blockSize(blocksize_x);
	dim3 gridSize;
	scan_range_temp = (scan_range_temp/16) + (scan_range_temp%16);
	gridSize.x = (scan_range_temp+ blockSize.x -1)/ blockSize.x;

	static int max_grid_size = 4096;
	if (gridSize.x > max_grid_size)
	{
		gridSize.x = max_grid_size;
	}

	Uint4 global_size = gridSize.x;
	global_size *= blocksize_x;

	slogfile.KernelStart();


	gpu_blastn_scan_Any<<< gridSize, blockSize >>>(
		cuda_scanAuxWrap.subject, 
		cuda_scanAuxWrap.offsetPairs, 
		cuda_scanAuxWrap.total_hits, 
		scan_range_temp, 
		scan_range[0],
		mask,
		shift,
		pv_array_bts,
		global_size,
		cuda_scanAuxWrap.lookupArray);

	getLastCudaError("gpu_blastn_scan_11_2mod4() execution failed.\n");

	slogfile.KernelEnd();
	slogfile.addTotalTime("Scan Kernel Time", slogfile.KernelElaplsedTime(), false);

	checkCudaErrors(cudaMemcpy(&total_hits, cuda_scanAuxWrap.total_hits, sizeof(Uint4), cudaMemcpyDeviceToHost));

	Int4 threadNum = 512;
	Int4 blockNum = (total_hits + threadNum - 1)/threadNum;
	dim3 gridDim(blockNum, 1);
	dim3 blockDim(threadNum, 1);

	//checkCudaErrors(cudaMemcpy(offset_pairs, d_offsetPairs, total_hits * sizeof(BlastOffsetPair), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemset(cuda_wrap.exact_hits_num, 0, sizeof(unsigned int)));  //初始化为0

	slogfile.KernelStart();

	kernel_lookupInBigHashTable<<<gridDim,blockDim>>>(
		cuda_wrap.hashtable,
		cuda_wrap.next_pos,
		total_hits,
		cuda_scanAuxWrap.offsetPairs,
		cuda_wrap.exact_offset_pairs,
		cuda_wrap.exact_hits_num
		);
	getLastCudaError("kernel_lookupInBigHashTable() execution failed.\n");
	checkCudaErrors(cudaMemcpy(&total_hits, cuda_wrap.exact_hits_num, sizeof(unsigned int), cudaMemcpyDeviceToHost));
	slogfile.KernelEnd();
	slogfile.addTotalTime("LookUpTableHash Time v1", slogfile.KernelElaplsedTime(), false);
	slogfile.addTotalNum("Kernel_lookupInBigHashTable hits", total_hits);

#if !GPU_EXT_RUN
	checkCudaErrors(cudaMemcpy(offset_pairs, cuda_wrap.exact_offset_pairs, total_hits * sizeof(BlastOffsetPair), cudaMemcpyDeviceToHost));
#endif
	//slogfile.m_file << "Scan hits:"<<total_hits<<endl;

	scan_range[0] = scan_range[1]+11;

	return total_hits;
}


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//extend
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
void s_initGPUMemoryforExtension( LookupTableWrap * lookup_wrap,
	BLAST_SequenceBlk * query )
{
	unsigned int * d_exact_hits_num;
	Int4 * d_hashtable, *d_next_pos;
	Uint1 * d_query;
	BlastOffsetPair *d_exact_offset_pairs;
	BlastMBLookupTable *lut = (BlastMBLookupTable *) lookup_wrap->lut;

	checkCudaErrors(cudaMalloc((void **) &d_exact_hits_num, sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc((void **) &d_hashtable, lut->hashsize * sizeof(Int4)));
	checkCudaErrors(cudaMalloc((void **) &d_next_pos, (query->length+1) * sizeof(Int4)));
	checkCudaErrors(cudaMalloc((void **) &d_query, query->length * sizeof(Uint1)));
	checkCudaErrors(cudaMalloc((void **) &d_exact_offset_pairs, OFFSET_ARRAY_SIZE * sizeof(BlastOffsetPair)));    //可能很小，具体不确定，仍按最大可能空间分配

	checkCudaErrors(cudaMemcpy(d_hashtable, lut->hashtable, sizeof(Int4) * lut->hashsize, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_next_pos, lut->next_pos, sizeof(Int4) * (query->length+1), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_query, query->sequence, sizeof(Uint1) * query->length, cudaMemcpyHostToDevice));

	cuda_wrap.exact_hits_num = d_exact_hits_num;
	cuda_wrap.hashtable = d_hashtable;
	cuda_wrap.next_pos = d_next_pos;
	cuda_wrap.query = d_query;
	cuda_wrap.exact_offset_pairs = d_exact_offset_pairs;
}

void  s_freeGPUMemoryforExtension()
{
	// release resources..
	checkCudaErrors(cudaFree(cuda_wrap.exact_hits_num));
	checkCudaErrors(cudaFree(cuda_wrap.hashtable));
	checkCudaErrors(cudaFree(cuda_wrap.next_pos));
	checkCudaErrors(cudaFree(cuda_wrap.query));
	checkCudaErrors(cudaFree(cuda_wrap.exact_offset_pairs));
}

/** Perform exact match extensions on the hits retrieved from
 * blastn/megablast lookup tables, assuming an arbitrary number of bases 
 * in a lookup and arbitrary start offset of each hit.
 * @param offset_pairs Array of query and subject offsets [in]
 * @param num_hits Size of the above arrays [in]
 * @param word_params Parameters for word extension [in]
 * @param lookup_wrap Lookup table wrapper structure [in]
 * @param query Query sequence data [in]
 * @param subject Subject sequence data [in]
 * @param matrix Scoring matrix for ungapped extension [in]
 * @param query_info Structure containing query context ranges [in]
 * @param ewp Word extension structure containing information about the 
 *            extent of already processed hits on each diagonal [in]
 * @param init_hitlist Structure to keep the extended hits. 
 *                     Must be allocated outside of this function [in] [out]
 * @param s_range The subject range [in]
 * @param cuda_wrap pointers to the the GPU device [in] 
 * @return Number of hits extended. 
 */
extern "C" Int4 
s_gpu_BlastNaExtend(const BlastOffsetPair * offset_pairs, Int4 num_hits,
                const BlastInitialWordParameters * word_params,
                LookupTableWrap * lookup_wrap,
                BLAST_SequenceBlk * query,
                BLAST_SequenceBlk * subject, Int4 ** matrix,
                BlastQueryInfo * query_info,
                Blast_ExtendWord * ewp,
                BlastInitHitList * init_hitlist,
                Uint4 s_range )
{
    Int4 hits_extended = 0;
    Int4 word_length, lut_word_length;
    BlastSeqLoc* masked_locations = NULL;
    BlastMBLookupTable *lut = (BlastMBLookupTable *) lookup_wrap->lut;
    word_length = lut->word_length;
    lut_word_length = lut->lut_word_length;
    masked_locations = lut->masked_locations;

	unsigned int exact_hits_num = 0;

	Int4 threadNum = 512;
	Int4 blockNum = (num_hits + threadNum - 1)/threadNum;
	dim3 gridDim(blockNum, 1);
	dim3 blockDim(threadNum, 1);

	//checkCudaErrors(cudaMemcpy(offset_pairs, d_offsetPairs, total_hits * sizeof(BlastOffsetPair), cudaMemcpyDeviceToHost));
	slogfile.KernelStart();
	checkCudaErrors(cudaMemset(cuda_wrap.exact_hits_num, 0, sizeof(unsigned int)));  //初始化为0
	kernel_s_BlastNaExtend<<<gridDim, blockDim>>> ( (Uint1*)cuda_scanAuxWrap.subject,
		                                             cuda_wrap.query,
													 cuda_wrap.hashtable,
													 cuda_wrap.next_pos,
													 word_length, lut_word_length,
													 num_hits,
													 cuda_scanAuxWrap.offsetPairs, cuda_wrap.exact_offset_pairs,
													 cuda_wrap.exact_hits_num, 
													 s_range );
	getLastCudaError("kernel_s_BlastNaExtend() execution failed.\n");
	checkCudaErrors(cudaMemcpy(&exact_hits_num, cuda_wrap.exact_hits_num, sizeof(unsigned int), cudaMemcpyDeviceToHost));

	slogfile.KernelEnd();
	slogfile.addTotalTime("kernel_s_BlastNaExtend", slogfile.KernelElaplsedTime(), false);

	checkCudaErrors(cudaMemcpy((void *)offset_pairs, cuda_wrap.exact_offset_pairs, exact_hits_num* sizeof(BlastOffsetPair), cudaMemcpyDeviceToHost));

	slogfile.addTotalNum("mini_extended hits", exact_hits_num, false);

	__int64 c1 = slogfile.Start();
	//printf("mini_extended_hits %d\n", exact_hits_num);
	for(int i=0; i<exact_hits_num; i++){
		Int4 s_offset = offset_pairs[i].qs_offsets.s_off;
		Int4 q_offset = offset_pairs[i].qs_offsets.q_off;
		//mini-extension之后的处理，仍然调用原有函数
/*		hits_extended += s_BlastnExtendInitialHit(query, subject, 

												  q_offset, s_offset,  
												  masked_locations, 
												  query_info, s_range, 
												  word_length, lut_word_length,
												  lookup_wrap,
												  word_params, matrix,
												  init_hitlist);*/

		hits_extended += s_BlastnDiagHashExtendInitialHit(query, subject, 
			q_offset, s_offset,  
			masked_locations, 
			query_info, s_range, 
			word_length, lut_word_length,
			lookup_wrap,
			word_params, matrix,
			ewp->hash_table,
			init_hitlist);
	}
	__int64 c2 = slogfile.End();
	slogfile.addTotalTime("Hits extend time",c1,c2, false);

	return hits_extended; 
}

extern "C" Int4 
	s_gpu_BlastNaExtend_v1(const BlastOffsetPair * offset_pairs, Int4 num_hits,
	const BlastInitialWordParameters * word_params,
	LookupTableWrap * lookup_wrap,
	BLAST_SequenceBlk * query,
	BLAST_SequenceBlk * subject, Int4 ** matrix,
	BlastQueryInfo * query_info,
	Blast_ExtendWord * ewp,
	BlastInitHitList * init_hitlist,
	Uint4 s_range )
{
	Int4 hits_extended = 0;
	Int4 word_length, lut_word_length;
	BlastSeqLoc* masked_locations = NULL;
	BlastMBLookupTable *lut = (BlastMBLookupTable *) lookup_wrap->lut;
	word_length = lut->word_length;
	lut_word_length = lut->lut_word_length;
	masked_locations = lut->masked_locations;

	unsigned int exact_hits_num = 0;

	Int4 threadNum = 512;
	Int4 blockNum = (num_hits + threadNum - 1)/threadNum;
	dim3 gridDim(blockNum, 1);
	dim3 blockDim(threadNum, 1);

	//checkCudaErrors(cudaMemcpy(offset_pairs, d_offsetPairs, total_hits * sizeof(BlastOffsetPair), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemset(cuda_wrap.exact_hits_num, 0, sizeof(unsigned int)));  //初始化为0

	slogfile.KernelStart();

	kernel_lookupInBigHashTable<<<gridDim,blockDim>>>(cuda_wrap.hashtable,
		cuda_wrap.next_pos,
		num_hits,
		cuda_scanAuxWrap.offsetPairs,
		cuda_wrap.exact_offset_pairs,
		cuda_wrap.exact_hits_num
		);
	getLastCudaError("kernel_lookupInBigHashTable() execution failed.\n");
	checkCudaErrors(cudaMemcpy(&exact_hits_num, cuda_wrap.exact_hits_num, sizeof(unsigned int), cudaMemcpyDeviceToHost));
	slogfile.KernelEnd();
	slogfile.addTotalTime("LookUpTableHash Time v1", slogfile.KernelElaplsedTime(), false );
	//printf("extended_hits %d,lookuped hash %d\n",num_hits, exact_hits_num);
	slogfile.addTotalNum("Kernel_lookupInBigHashTable hits", exact_hits_num, false);

	Int4 blockNum_Ex = (exact_hits_num + threadNum - 1)/threadNum;


	dim3 gridDim_Ex(blockNum_Ex, 1);
	dim3 blockDim_Ex(threadNum, 1);

	checkCudaErrors(cudaMemset(cuda_wrap.exact_hits_num, 0, sizeof(unsigned int)));  //初始化为0
	Int4 ext_to =  word_length - lut_word_length;

	slogfile.KernelStart();
	kernel_s_BlastNaExtend_withoutHash<<<gridDim_Ex, blockDim_Ex>>> ( (Uint1*)cuda_scanAuxWrap.subject,
		cuda_wrap.query,
		lut_word_length,
		ext_to,		
		exact_hits_num,
		cuda_wrap.exact_offset_pairs, cuda_scanAuxWrap.offsetPairs, 
		cuda_wrap.exact_hits_num, 
		s_range);

	getLastCudaError("kernel_s_BlastNaExtend_withoutHash() v1 execution failed.\n");

	checkCudaErrors(cudaMemcpy(&exact_hits_num, cuda_wrap.exact_hits_num, sizeof(unsigned int), cudaMemcpyDeviceToHost));
	slogfile.KernelEnd();
	slogfile.addTotalTime("kernel_s_BlastNaExtend_withoutHash Time v1", slogfile.KernelElaplsedTime(), false);

	//printf("mini_extended_hits %d\n", exact_hits_num);
	slogfile.addTotalNum("mini_extended hits", exact_hits_num,false);

	if (exact_hits_num >0)
	{
		slogfile.Start();
		checkCudaErrors(cudaMemcpy((void *)offset_pairs, cuda_scanAuxWrap.offsetPairs, exact_hits_num* sizeof(BlastOffsetPair), cudaMemcpyDeviceToHost));	 	
		slogfile.End();
		slogfile.addTotalTime("GPU->CPU memory Time", slogfile.elaplsedTime(), false);
	}

	__int64 c1 = slogfile.Start();
	if (exact_hits_num >0)
	{  
#if USE_OPENMP
#pragma omp parallel for reduction(+:hits_extended)
#endif
		for(int i=0; i<exact_hits_num; i++){
			Int4 s_offset = offset_pairs[i].qs_offsets.s_off;
			Int4 q_offset = offset_pairs[i].qs_offsets.q_off;
			//mini-extension之后的处理，仍然调用原有函数
		//	hits_extended += s_BlastnExtendInitialHit(query, subject, 
		//		q_offset, s_offset,  
		//		masked_locations, 
		//		query_info, s_range, 
		//		word_length, lut_word_length,
		//		lookup_wrap,
		//		word_params, matrix,
		//		init_hitlist);  
		//}
			hits_extended += s_BlastnDiagHashExtendInitialHit(query, subject, 
				q_offset, s_offset,  
				masked_locations, 
				query_info, s_range, 
				word_length, lut_word_length,
				lookup_wrap,
				word_params, matrix,
				ewp->hash_table,
				init_hitlist);
		}
	}


	__int64 c2 = slogfile.End();
	slogfile.addTotalTime("Hits extend time",c1,c2, false);

	return hits_extended; 
}

extern "C" Int4 
	s_gpu_BlastNaExtend_v2(const BlastOffsetPair * offset_pairs, Int4 num_hits,
	const BlastInitialWordParameters * word_params,
	LookupTableWrap * lookup_wrap,
	BLAST_SequenceBlk * query,
	BLAST_SequenceBlk * subject, Int4 ** matrix,
	BlastQueryInfo * query_info,
	Blast_ExtendWord * ewp,
	BlastInitHitList * init_hitlist,
	Uint4 s_range )
{
	Int4 hits_extended = 0;
	Int4 word_length, lut_word_length;
	BlastSeqLoc* masked_locations = NULL;
	BlastMBLookupTable *lut = (BlastMBLookupTable *) lookup_wrap->lut;
	word_length = lut->word_length;
	lut_word_length = lut->lut_word_length;
	masked_locations = lut->masked_locations;

	unsigned int exact_hits_num = 0;

	Int4 threadNum = 256;
	Int4 blockNum_Ex = (num_hits + threadNum - 1)/threadNum;
	dim3 gridDim_Ex(blockNum_Ex, 1);
	dim3 blockDim_Ex(threadNum, 1);

	//printf("%d, %d\n", threadNum, blockNum_Ex);

	checkCudaErrors(cudaMemset(cuda_wrap.exact_hits_num, 0, sizeof(unsigned int)));  //初始化为0

	Int4 ext_to =  word_length - lut_word_length;

	slogfile.KernelStart();

	kernel_s_BlastNaExtend_withoutHash<<<gridDim_Ex, blockDim_Ex>>> ( (Uint1*)cuda_scanAuxWrap.subject,
		cuda_wrap.query,
		lut_word_length,
		ext_to,		
		num_hits,
		cuda_wrap.exact_offset_pairs, cuda_scanAuxWrap.offsetPairs, 
		cuda_wrap.exact_hits_num, 
		s_range );
	getLastCudaError("kernel_s_BlastNaExtend_withoutHash() v1 execution failed.\n");

	checkCudaErrors(cudaMemcpy(&exact_hits_num, cuda_wrap.exact_hits_num, sizeof(unsigned int), cudaMemcpyDeviceToHost));
	slogfile.KernelEnd();
	slogfile.addTotalTime("kernel_s_BlastNaExtend_withoutHash Time v1", slogfile.KernelElaplsedTime(),false);
	slogfile.addTotalNum("mini_extended hits", exact_hits_num,false);

	if (exact_hits_num >0)
	{
		slogfile.KernelStart();
		checkCudaErrors(cudaMemcpy((void *)offset_pairs, cuda_scanAuxWrap.offsetPairs, exact_hits_num* sizeof(BlastOffsetPair), cudaMemcpyDeviceToHost));	 	
		slogfile.KernelEnd();
		slogfile.addTotalTime("GPU->CPU memory Time", slogfile.KernelElaplsedTime(),false);
	}

	__int64 c1 = slogfile.Start();
	if (exact_hits_num >0)
	{  
#if USE_OPENMP
#pragma omp parallel for reduction(+:hits_extended)
#endif
		for(int i=0; i<exact_hits_num; i++){
			Int4 s_offset = offset_pairs[i].qs_offsets.s_off;
			Int4 q_offset = offset_pairs[i].qs_offsets.q_off;
			hits_extended += s_BlastnDiagHashExtendInitialHit(query, subject, 
				q_offset, s_offset,  
				masked_locations, 
				query_info, s_range, 
				word_length, lut_word_length,
				lookup_wrap,
				word_params, matrix,
				ewp->hash_table,
				init_hitlist);
		}
	}


	__int64 c2 = slogfile.End();
	slogfile.addTotalTime("Hits extend time",c1,c2, false);

	return hits_extended; 
}

extern "C" void initAllGPUMemory( 
	LookupTableWrap* lookup_wrap,
	BLAST_SequenceBlk* query)
{
	if (!is_initGPUMem)
	{
		s_initScanGPUMemory(lookup_wrap);
		s_initGPUMemoryforExtension(lookup_wrap,query);
	}
	is_initGPUMem = true;
}
extern "C" void freeAllGPUMemory()   //gpu_blastn added by kyzhao
{
	if (is_initGPUMem)
	{
		s_freeScanGPUMemory();
		s_freeGPUMemoryforExtension();
	}
	is_initGPUMem = false;
}

//////////////////////////////////////////////////////////////////////////
//
void s_initScanGPUMemory(const BlastSeqSrc* seq_src, LookupTableWrap* lookup_wrap)
{
	BlastOffsetPair* d_offsetPairs = NULL;
	Uint4* d_subject = NULL;
	Uint4* d_total_hits = NULL;

	checkCudaErrors(cudaMalloc((void **)&d_total_hits,sizeof(Uint4)));
	checkCudaErrors(cudaMalloc((void **)&d_subject, MAX_DBSEQ_LEN*sizeof(Uint4)/4));
	checkCudaErrors(cudaMalloc((void **)&d_offsetPairs, OFFSET_ARRAY_SIZE * sizeof(BlastOffsetPair)));

	cuda_scanAuxWrap.offsetPairs = d_offsetPairs;
	cuda_scanAuxWrap.subject = d_subject;
	cuda_scanAuxWrap.total_hits = d_total_hits;
	
	BlastMBLookupTable* mb_lt = (BlastMBLookupTable*) lookup_wrap->lut;
	Int4 pv_size = mb_lt->hashsize/(1<<mb_lt->pv_array_bts);
	s_initLookupTableTexture(pv_size, mb_lt->pv_array);
}

// each volume subject process in one time.
extern "C" void initAllGPUMemoryVolume(const BlastSeqSrc* seq_src, LookupTableWrap* lookup_wrap, BLAST_SequenceBlk* query)
{
	if (!is_initGPUMem)
	{
		s_initScanGPUMemory(seq_src, lookup_wrap);
		s_initGPUMemoryforExtension(lookup_wrap,query);
	}
	is_initGPUMem = true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//multi queries
//////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
typedef struct cudaScanAuxWrapMultiQueries 
{
	BlastOffsetPair* offsetPairs;
	BlastOffsetPair* over_offset_pairs;
	Uint4** subject;
	Uint4   subject_id;
	Uint4 *total_hits;
	Uint4 *over_hits_num;
} cudaScanAuxWrapMultiQueries;

/** Wrapper structure for different types of cuda structures */
typedef struct cudaMBHashAuxWrap {
	PV_ARRAY_TYPE* lookupArray;
	Int4 * hashtable;
	Int4 * next_pos;
	Int4 next_pos_len;
	Uint1 * query;
} cudaMBHashAuxWrap;


static cudaScanAuxWrapMultiQueries cuda_scanMultiAuxWrap;
static cudaMBHashAuxWrap cuda_MBHashWrap;

bool isInitScanGPUMem = false;
bool isInitExternGPUMem = false;

void InitScanGPUMem(const BlastSeqSrc* seq_src)
{
	Uint4* d_total_hits = NULL;
	Uint4* d_over_hits_num = NULL;
	BlastOffsetPair* d_offsetPairs = NULL;
	BlastOffsetPair* d_over_offsetPairs = NULL;
	int subject_num = BlastSeqSrcGetNumSeqs(seq_src);
	Uint4** d_subject = new Uint4*[subject_num]();	

	checkCudaErrors(cudaMalloc((void **)&d_total_hits,sizeof(Uint4)));
	checkCudaErrors(cudaMalloc((void **)&d_over_hits_num,sizeof(Uint4)));
	checkCudaErrors(cudaMalloc((void **)&d_offsetPairs, OFFSET_ARRAY_SIZE * sizeof(BlastOffsetPair)));
	checkCudaErrors(cudaMalloc((void **)&d_over_offsetPairs, OFFSET_ARRAY_SIZE * sizeof(BlastOffsetPair)));

	cuda_scanMultiAuxWrap.total_hits = d_total_hits;
	cuda_scanMultiAuxWrap.over_hits_num = d_over_hits_num;
	cuda_scanMultiAuxWrap.offsetPairs = d_offsetPairs;
	cuda_scanMultiAuxWrap.over_offset_pairs = d_over_offsetPairs;	
	cuda_scanMultiAuxWrap.subject_id = 0;
	
	cuda_scanMultiAuxWrap.subject = d_subject;
	for (int i = 0; i < subject_num; i ++)
	{
		cuda_scanMultiAuxWrap.subject[i]= NULL;
	}
}

extern "C" void InitGPUMemMultiSeq(const BlastSeqSrc* seq_src)
{
	if (isInitScanGPUMem == false)
	{
		InitScanGPUMem(seq_src);
	}
	isInitScanGPUMem = true;
}

extern "C" void ReleaseGPUMemMultiSeq()
{
	if (isInitScanGPUMem == true)
	{
		checkCudaErrors(cudaFree(cuda_scanMultiAuxWrap.total_hits));
		checkCudaErrors(cudaFree(cuda_scanMultiAuxWrap.over_hits_num));
		checkCudaErrors(cudaFree(cuda_scanMultiAuxWrap.offsetPairs));
		checkCudaErrors(cudaFree(cuda_scanMultiAuxWrap.over_offset_pairs));

		for ( int i = 0; i < cuda_scanMultiAuxWrap.subject_id; i++)
		{
			if ( cuda_scanMultiAuxWrap.subject[i] != NULL)
			{
				checkCudaErrors(cudaFree(cuda_scanMultiAuxWrap.subject[i]));
			}
		}
		delete[] cuda_scanMultiAuxWrap.subject;
	}
	isInitScanGPUMem = false;
}

extern "C" void SetSubjectID(int id)
{
	cuda_scanMultiAuxWrap.subject_id = id;
}

//////////////////////////////////////////////////////////////////////////
//extend
//////////////////////////////////////////////////////////////////////////
void InitLookupTableTexture(LookupTableWrap* lookup_wrap)
{
	BlastMBLookupTable* mb_lt = (BlastMBLookupTable*) lookup_wrap->lut;
	Int4 pv_size = mb_lt->hashsize/(1<<mb_lt->pv_array_bts);

	PV_ARRAY_TYPE* d_lookupArray = NULL;
	//create cuda array
	checkCudaErrors(cudaMalloc( (void**) &d_lookupArray, pv_size *sizeof(PV_ARRAY_TYPE)));
	checkCudaErrors(cudaMemcpy( d_lookupArray, mb_lt->pv_array, pv_size * sizeof (PV_ARRAY_TYPE), cudaMemcpyHostToDevice));
	// Bind the array to the texture
	SET_PVARRAY_BASE;

	cuda_MBHashWrap.lookupArray = d_lookupArray;
}

void FreeLookupTableTexture()
{
	checkCudaErrors(cudaUnbindTexture(tx_pv_array));
	checkCudaErrors(cudaFree(cuda_MBHashWrap.lookupArray));
}

extern "C" void InitGPUMemforExtension(LookupTableWrap * lookup_wrap,	BLAST_SequenceBlk * query)
{
	if (isInitExternGPUMem == false)
	{
		unsigned int * d_exact_hits_num;
		Int4 * d_hashtable, *d_next_pos;
		Uint1 * d_query;
		BlastOffsetPair *d_exact_offset_pairs;
		BlastMBLookupTable *lut = (BlastMBLookupTable *) lookup_wrap->lut;

		Uint4 biggest_hashSize = lut->hashsize;
		Uint4 biggest_length_next = query->length+1;
		Uint4 biggest_length	  = query->length;

		checkCudaErrors(cudaMalloc((void **) &d_hashtable, biggest_hashSize * sizeof(Int4))); // lut->hashsize for biggeest size
		checkCudaErrors(cudaMalloc((void **) &d_next_pos, biggest_length_next * sizeof(Int4)));  // (query->length+1) for biggest size
		checkCudaErrors(cudaMalloc((void **) &d_query, biggest_length * sizeof(Uint1)));	 // query->length for biggest size

		checkCudaErrors(cudaMemcpy(d_hashtable, lut->hashtable, sizeof(Int4) * biggest_hashSize, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_next_pos, lut->next_pos, sizeof(Int4) * biggest_length_next, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_query, query->sequence, sizeof(Uint1) * biggest_length, cudaMemcpyHostToDevice));

		cuda_MBHashWrap.hashtable = d_hashtable;
		cuda_MBHashWrap.next_pos = d_next_pos;
		cuda_MBHashWrap.query = d_query;
		cuda_MBHashWrap.next_pos_len = biggest_length_next;

		InitLookupTableTexture(lookup_wrap);
	}
	isInitExternGPUMem = true;
}

extern "C" void ReleaseExtenGPUMem()
{
	if (isInitExternGPUMem == true)
	{
		checkCudaErrors(cudaFree(cuda_MBHashWrap.hashtable));
		checkCudaErrors(cudaFree(cuda_MBHashWrap.next_pos));
		checkCudaErrors(cudaFree(cuda_MBHashWrap.query));
		
		FreeLookupTableTexture();
	}
	isInitExternGPUMem = false;
}

/** Scan the compressed subject sequence, returning 11-letter word hits
* with stride two plus a multiple of four. Assumes a megablast lookup table
* @param lookup_wrap Pointer to the (wrapper to) lookup table [in]
* @param subject The (compressed) sequence to be scanned for words [in]
* @param offset_pairs Array of query and subject positions where words are 
*                found [out]
* @param max_hits The allocated size of the above array - how many offsets 
*        can be returned [in]
* @param scan_range The starting and ending pos to be scanned [in] 
*        on exit, scan_range[0] is updated to be the stopping pos [out]
*/
extern "C" Int4 
	s_gpu_MBScanSubject_11_2Mod4_v1_scankernel_Opt(const LookupTableWrap* lookup_wrap,
	const BLAST_SequenceBlk* subject,
	BlastOffsetPair* NCBI_RESTRICT offset_pairs, Int4 max_hits,  
	Int4* scan_range)
{
	BlastMBLookupTable* mb_lt = (BlastMBLookupTable*) lookup_wrap->lut;
	Uint4 total_hits = 0;
	Int4 top_shift; 

	max_hits -= mb_lt->longest_chain;
	ASSERT(lookup_wrap->lut_type == eMBLookupTable);
	ASSERT(mb_lt->lut_word_length == 11);
	if(scan_range[0] > scan_range[1]) return 0;

	top_shift = 2;
	Int4 pv_array_bts = mb_lt->pv_array_bts;
	Uint4 scan_range_temp = (scan_range[1]+mb_lt->lut_word_length - scan_range[0]);
	
	Uint4 subject_len = subject->length;

	slogfile.Start();
	if (cuda_scanMultiAuxWrap.subject[cuda_scanMultiAuxWrap.subject_id] == NULL)
	{
		//printf("id:%d\n", cuda_scanMultiAuxWrap.subject_id);
		checkCudaErrors(cudaMalloc((void **)&cuda_scanMultiAuxWrap.subject[cuda_scanMultiAuxWrap.subject_id],((subject_len) +3)/4));
		checkCudaErrors(cudaMemcpy(cuda_scanMultiAuxWrap.subject[cuda_scanMultiAuxWrap.subject_id], subject->sequence, (subject_len)/4 , cudaMemcpyHostToDevice));
	}
	slogfile.End();
	slogfile.addTotalTime("Scan CPU -> GPU Memory Time",slogfile.elaplsedTime(),false);
	 	
	checkCudaErrors(cudaMemset(cuda_scanMultiAuxWrap.total_hits, 0, sizeof(unsigned int)));  //初始化为0
	
	static int blocksize_x = 128;
	dim3 blockSize(blocksize_x);
	dim3 gridSize;
	//scan_range_temp = (scan_range_temp/16) + (scan_range_temp%16);
	scan_range_temp = scan_range_temp/16;
	gridSize.x = (scan_range_temp+ blockSize.x -1)/ blockSize.x;

	static int max_grid_size = 4096;
	if (gridSize.x > max_grid_size)
	{
		gridSize.x = max_grid_size;
	}

	Uint4 global_size = gridSize.x;
	global_size *= blocksize_x;

	slogfile.KernelStart();

	//printf("%d %d\n", gridSize.x, blockSize.x);

	gpu_blastn_scan_11_2mod4_v1<<< gridSize, blockSize >>>(
		cuda_scanMultiAuxWrap.subject[cuda_scanMultiAuxWrap.subject_id], 
		cuda_scanMultiAuxWrap.offsetPairs, 
		cuda_scanMultiAuxWrap.total_hits, 
		scan_range_temp, 
		scan_range[0], 
		top_shift, 
		pv_array_bts,
		global_size,
		cuda_MBHashWrap.lookupArray); 

	getLastCudaError("gpu_blastn_scan_11_2mod4() execution failed.\n");

	slogfile.KernelEnd();
	slogfile.addTotalTime("Scan Kernel Time", slogfile.KernelElaplsedTime(),false);

	checkCudaErrors(cudaMemcpy(&total_hits, cuda_scanMultiAuxWrap.total_hits, sizeof(Uint4), cudaMemcpyDeviceToHost));

	//printf("total_hits:%d\n",total_hits);
	//total_hits = 0;

	Int4 threadNum = 512;
	Int4 blockNum = (total_hits + threadNum - 1)/threadNum;
	dim3 gridDim(blockNum, 1);
	dim3 blockDim(threadNum, 1);

	checkCudaErrors(cudaMemset(cuda_scanMultiAuxWrap.over_hits_num, 0, sizeof(unsigned int)));  //初始化为0

	slogfile.KernelStart();

	kernel_lookupInBigHashTable_v1<<<gridDim,blockDim>>>(
		cuda_MBHashWrap.hashtable,
		cuda_MBHashWrap.next_pos,
		total_hits,
		cuda_scanMultiAuxWrap.offsetPairs,
		cuda_scanMultiAuxWrap.over_offset_pairs,
		cuda_scanMultiAuxWrap.over_hits_num,
		cuda_MBHashWrap.next_pos_len
		);
	getLastCudaError("kernel_lookupInBigHashTable() execution failed.\n");
	slogfile.KernelEnd();
	slogfile.addTotalTime("LookUpTableHash Time v1", slogfile.KernelElaplsedTime(), false );

	checkCudaErrors(cudaMemcpy(&total_hits, cuda_scanMultiAuxWrap.over_hits_num, sizeof(unsigned int), cudaMemcpyDeviceToHost));
	slogfile.addTotalNum("Kernel_lookupInBigHashTable hits", total_hits, false);
	//printf("bighash_hits:%d\n", total_hits);

#if !GPU_EXT_RUN
	checkCudaErrors(cudaMemcpy(offset_pairs, cuda_scanMultiAuxWrap.over_offset_pairs, total_hits * sizeof(BlastOffsetPair), cudaMemcpyDeviceToHost));
#endif
	//slogfile.m_file << "Scan hits:"<<total_hits<<endl;

	scan_range[0] = scan_range[1]+mb_lt->lut_word_length;

	return total_hits;
}


extern "C" Int4 
	s_gpu_MBScanSubject_Any_scankernel_Opt(const LookupTableWrap* lookup_wrap,
	const BLAST_SequenceBlk* subject,
	BlastOffsetPair* NCBI_RESTRICT offset_pairs, Int4 max_hits,  
	Int4* scan_range)
{
	BlastMBLookupTable* mb_lt = (BlastMBLookupTable*) lookup_wrap->lut;
	Uint4 total_hits = 0;
	Int4 shift; 
	Int4 mask = mb_lt->hashsize - 1;
	max_hits -= mb_lt->longest_chain;
	ASSERT(lookup_wrap->lut_type == eMBLookupTable);
	ASSERT(mb_lt->lut_word_length == 11||
		mb_lt->lut_word_length == 12);
	if(scan_range[0] > scan_range[1]) return 0;

	shift = 2*(16 - (scan_range[0] % COMPRESSION_RATIO + mb_lt->lut_word_length));
	Uint4 pv_array_bts = mb_lt->pv_array_bts;
	
	Uint4 scan_range_temp = (scan_range[1] - scan_range[0]);

	Uint4 subject_len = subject->length; //length is bp, 2bit.

	slogfile.Start();
	if (cuda_scanMultiAuxWrap.subject[cuda_scanMultiAuxWrap.subject_id] == NULL)
	{
		//printf("id:%d\n", cuda_scanMultiAuxWrap.subject_id);
		checkCudaErrors(cudaMalloc((void **)&cuda_scanMultiAuxWrap.subject[cuda_scanMultiAuxWrap.subject_id],(subject_len+3)/4));
		checkCudaErrors(cudaMemcpy(cuda_scanMultiAuxWrap.subject[cuda_scanMultiAuxWrap.subject_id], subject->sequence, (subject_len)/4 , cudaMemcpyHostToDevice));
	}
	checkCudaErrors(cudaMemset(cuda_scanMultiAuxWrap.total_hits, 0, sizeof(unsigned int)));  //初始化为0

	slogfile.End();
	slogfile.addTotalTime("Scan CPU -> GPU Memory Time",slogfile.elaplsedTime(),false);

	static int blocksize_x = 128;
	dim3 blockSize(blocksize_x);
	dim3 gridSize;
	scan_range_temp = scan_range_temp/16;
	gridSize.x = (scan_range_temp+ blockSize.x -1)/ blockSize.x;

	static int max_grid_size = 4096;
	if (gridSize.x > max_grid_size)
	{
		gridSize.x = max_grid_size;
	}
	Uint4 global_size = gridSize.x;
	global_size *= blocksize_x;

	slogfile.KernelStart();
	gpu_blastn_scan_Any<<< gridSize, blockSize >>>(
		cuda_scanMultiAuxWrap.subject[cuda_scanMultiAuxWrap.subject_id], 
		cuda_scanMultiAuxWrap.offsetPairs, 
		cuda_scanMultiAuxWrap.total_hits, 
		scan_range_temp, 
		scan_range[0],
		mask,
		shift,
		pv_array_bts,
		global_size,
		cuda_MBHashWrap.lookupArray);
	getLastCudaError("gpu_blastn_scan_11_2mod4() execution failed.\n");
	slogfile.KernelEnd();
	slogfile.addTotalTime("Scan Kernel Time", slogfile.KernelElaplsedTime(),false);

	checkCudaErrors(cudaMemcpy(&total_hits, cuda_scanMultiAuxWrap.total_hits, sizeof(Uint4), cudaMemcpyDeviceToHost));

	Int4 threadNum = 512;
	Int4 blockNum = (total_hits + threadNum - 1)/threadNum;
	dim3 gridDim(blockNum, 1);
	dim3 blockDim(threadNum, 1); 
	//checkCudaErrors(cudaMemcpy(offset_pairs, d_offsetPairs, total_hits * sizeof(BlastOffsetPair), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemset(cuda_scanMultiAuxWrap.over_hits_num, 0, sizeof(unsigned int)));  //初始化为0

	slogfile.KernelStart();	 
	kernel_lookupInBigHashTable_v1<<<gridDim,blockDim>>>(
		cuda_MBHashWrap.hashtable,
		cuda_MBHashWrap.next_pos,
		total_hits,
		cuda_scanMultiAuxWrap.offsetPairs,
		cuda_scanMultiAuxWrap.over_offset_pairs,
		cuda_scanMultiAuxWrap.over_hits_num,
		cuda_MBHashWrap.next_pos_len
		);

	getLastCudaError("kernel_lookupInBigHashTable() execution failed.\n");
	slogfile.KernelEnd();
	slogfile.addTotalTime("LookUpTableHash Time v1", slogfile.KernelElaplsedTime(), false );
	checkCudaErrors(cudaMemcpy(&total_hits, cuda_scanMultiAuxWrap.over_hits_num, sizeof(unsigned int), cudaMemcpyDeviceToHost));
	slogfile.addTotalNum("Kernel_lookupInBigHashTable hits", total_hits, false);

#if !GPU_EXT_RUN
	checkCudaErrors(cudaMemcpy(offset_pairs, cuda_scanMultiAuxWrap.over_offset_pairs, total_hits * sizeof(BlastOffsetPair), cudaMemcpyDeviceToHost));
#endif
	//slogfile.m_file << "Scan hits:"<<total_hits<<endl;

	scan_range[0] = scan_range[1]+mb_lt->lut_word_length;

	return total_hits;
}

//////////////////////////////////////////////////////////////////////////
///MiniExtend
//////////////////////////////////////////////////////////////////////////

extern "C" Int4 
	s_gpu_BlastNaExtend_Opt(BlastOffsetPair * offset_pairs, Int4 num_hits,
	const BlastInitialWordParameters * word_params,
	LookupTableWrap * lookup_wrap,
	BLAST_SequenceBlk * query,
	BLAST_SequenceBlk * subject, Int4 ** matrix,
	BlastQueryInfo * query_info,
	Blast_ExtendWord * ewp,
	BlastInitHitList * init_hitlist,
	Uint4 s_range )
{
	Int4 hits_extended = 0;
	Int4 word_length, lut_word_length;
	BlastSeqLoc* masked_locations = NULL;
	BlastMBLookupTable *lut = (BlastMBLookupTable *) lookup_wrap->lut;
	word_length = lut->word_length;
	lut_word_length = lut->lut_word_length;
	masked_locations = lut->masked_locations;

	unsigned int exact_hits_num = 0;

	Int4 threadNum = 256;
	Int4 blockNum_Ex = (num_hits + threadNum - 1)/threadNum;
	dim3 gridDim_Ex(blockNum_Ex, 1);
	dim3 blockDim_Ex(threadNum, 1);

	//printf("%d, %d\n", threadNum, blockNum_Ex);

	checkCudaErrors(cudaMemset(cuda_scanMultiAuxWrap.over_hits_num, 0, sizeof(unsigned int)));  //初始化为0

	Int4 ext_to =  word_length - lut_word_length;

	//printf("id:%d,num_his:%d \n", cuda_scanMultiAuxWrap.subject_id,num_hits);
	//printf("query length:%d\n", query->length);

	slogfile.KernelStart();

 	kernel_s_BlastNaExtend_withoutHash_v1<<<gridDim_Ex, blockDim_Ex>>> (
		(Uint1*)cuda_scanMultiAuxWrap.subject[cuda_scanMultiAuxWrap.subject_id],
		cuda_MBHashWrap.query,
		lut_word_length,
		ext_to,		
		num_hits,
		cuda_scanMultiAuxWrap.over_offset_pairs, cuda_scanMultiAuxWrap.offsetPairs, 
		cuda_scanMultiAuxWrap.over_hits_num, 
		s_range,
		query->length);

	getLastCudaError("kernel_s_BlastNaExtend_withoutHash() v1 execution failed.\n");

	slogfile.KernelEnd();
	slogfile.addTotalTime("kernel_s_BlastNaExtend_withoutHash Time v1", slogfile.KernelElaplsedTime(), false);
	
	checkCudaErrors(cudaMemcpy(&exact_hits_num, cuda_scanMultiAuxWrap.over_hits_num, sizeof(unsigned int), cudaMemcpyDeviceToHost));
	slogfile.addTotalNum("mini_extended hits", exact_hits_num,false);

	//printf("hits:%d\n", exact_hits_num);
	//exact_hits_num =0;

	if (exact_hits_num >0)
	{
		slogfile.Start();
		checkCudaErrors(cudaMemcpy(offset_pairs, cuda_scanMultiAuxWrap.offsetPairs, exact_hits_num* sizeof(BlastOffsetPair), cudaMemcpyDeviceToHost));	 	
		slogfile.End();
		slogfile.addTotalTime("GPU->CPU memory Time", slogfile.elaplsedTime(), false);
	}

	__int64 c1 = slogfile.Start();
	if (exact_hits_num >0)
	{  
#if USE_OPENMP
#pragma omp parallel for reduction(+:hits_extended)
#endif
		for(int i=0; i<exact_hits_num; i++){
			Int4 s_offset = offset_pairs[i].qs_offsets.s_off;
			Int4 q_offset = offset_pairs[i].qs_offsets.q_off;
			hits_extended += s_BlastnDiagHashExtendInitialHit(query, subject, 
				q_offset, s_offset,  
				masked_locations, 
				query_info, s_range, 
				word_length, lut_word_length,
				lookup_wrap,
				word_params, matrix,
				ewp->hash_table,
				init_hitlist);
		}
	}


	__int64 c2 = slogfile.End();
	slogfile.addTotalTime("Hits extend time",c1,c2, false);

	return hits_extended; 
}

