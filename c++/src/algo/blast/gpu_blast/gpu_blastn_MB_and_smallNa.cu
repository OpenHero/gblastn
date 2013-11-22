#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h> // for set

#include <algo/blast/core/blast_nalookup.h>
#include <algo/blast/core/blast_nascan.h>
#include <algo/blast/core/blast_util.h>
#include <algo/blast/core/lookup_wrap.h>         //for MAX_DBSEQ_LEN
#include <algo/blast/core/blast_gapalign.h>      //for OFFSET_ARRAY_SIZE  

#include "gpu_blastn_ungapped_extension_functions.h"

#include "gpu_blastn_mb_scan_kernel_v3.cuh"
#include "gpu_blastn_lookup_hash_kernel_v3.cuh"
#include "gpu_blastn_mini_extension_kernel_v3.cuh"

#include "gpu_blastn_small_scan_kernel_v3.cuh"
#include "gpu_blastn_small_lookuptable_kernel_v3.cuh"
#include "gpu_blastn_small_mini_extension_kernel_v3.cuh"


#include <algo/blast/gpu_blast/gpu_logfile.h>
#include <algo/blast/gpu_blast/gpu_blast_multi_gpu_utils.hpp>

#include <iostream>
using namespace std;


struct cudaScanAuxWrapMultiQueries : public GpuObject  
{
	BlastOffsetPair* offsetPairs;
	BlastOffsetPair* over_offset_pairs;
	Uint4** subject;
	Uint4   subject_id;
	Uint4 *total_hits;
	Uint4 *over_hits_num;
	Uint4 *counter;
	Uint4 *zero;
};

struct OffsetPairCmp {
	__host__ __device__ 
		bool operator()(const BlastOffsetPair& bop1,const BlastOffsetPair& bop2)
	{
		return bop1.qs_offsets.s_off < bop2.qs_offsets.s_off;
	}
};

//////////////////////////////////////////////////////////////////////////
//multi queries
//////////////////////////////////////////////////////////////////////////

//static cudaScanAuxWrapMultiQueries cuda_scanMultiDBAuxWrap;

//////////////////////////////////////////////////////////////////////////
//init database gpu memory
void InitGPU_DB_Mem(int subject_seq_num, int max_len)
{
	GpuData* gpu_obj = BlastMGPUUtil.GetCurrentThreadGPUData();

	if (gpu_obj->m_global == NULL)
	{
		Uint4* d_total_hits = NULL;
		Uint4* d_over_hits_num = NULL;
		BlastOffsetPair* d_offsetPairs = NULL;
		BlastOffsetPair* d_over_offsetPairs = NULL;
		int subject_num = subject_seq_num;
		Uint4** d_subject = new Uint4*[subject_num]();
		
		Uint4* d_counter = NULL;
		Uint4* d_zero = NULL;

		checkCudaErrors(cudaMalloc((void **)&d_total_hits,sizeof(Uint4)));
		checkCudaErrors(cudaMalloc((void **)&d_over_hits_num,sizeof(Uint4)));
		checkCudaErrors(cudaMalloc((void **)&d_offsetPairs, max_len * sizeof(BlastOffsetPair)));
		checkCudaErrors(cudaMalloc((void **)&d_over_offsetPairs, max_len * sizeof(BlastOffsetPair)));
		checkCudaErrors(cudaMalloc((void **)&d_counter,3*sizeof(Uint4)));
		checkCudaErrors(cudaMalloc((void **)&d_zero,3*sizeof(Uint4)));

		checkCudaErrors(cudaMemset(d_zero,0,3*sizeof(Uint4)));

		cudaScanAuxWrapMultiQueries * p_scanMultiDBAuxWrap = new cudaScanAuxWrapMultiQueries();
		p_scanMultiDBAuxWrap->total_hits = d_total_hits;
		p_scanMultiDBAuxWrap->over_hits_num = d_over_hits_num;
		p_scanMultiDBAuxWrap->offsetPairs = d_offsetPairs;
		p_scanMultiDBAuxWrap->over_offset_pairs = d_over_offsetPairs;	
		p_scanMultiDBAuxWrap->subject_id = 0;
		
		p_scanMultiDBAuxWrap->counter = d_counter;
		p_scanMultiDBAuxWrap->zero = d_zero;

		p_scanMultiDBAuxWrap->subject = d_subject;
		for (int i = 0; i < subject_num; i ++)
		{
			p_scanMultiDBAuxWrap->subject[i]= NULL;
		}

		gpu_obj->m_global = p_scanMultiDBAuxWrap;
	}
}

void InitGPUMem_DB_MultiSeq(int subject_seq_num, int max_len)
{
	InitGPU_DB_Mem(subject_seq_num, max_len);
}

void ReleaseGPUMem_DB_MultiSeq()
{
	GpuData* gpu_obj = BlastMGPUUtil.GetCurrentThreadGPUData();
	if (gpu_obj->m_global != NULL)
	{
		cudaScanAuxWrapMultiQueries * p_scanMultiDBAuxWrap = (cudaScanAuxWrapMultiQueries *) gpu_obj->m_global;

		checkCudaErrors(cudaFree(p_scanMultiDBAuxWrap->total_hits));
		checkCudaErrors(cudaFree(p_scanMultiDBAuxWrap->over_hits_num));
		checkCudaErrors(cudaFree(p_scanMultiDBAuxWrap->offsetPairs));
		checkCudaErrors(cudaFree(p_scanMultiDBAuxWrap->over_offset_pairs));

		checkCudaErrors(cudaFree(p_scanMultiDBAuxWrap->counter));
		checkCudaErrors(cudaFree(p_scanMultiDBAuxWrap->zero));

		for ( int i = 0; i < p_scanMultiDBAuxWrap->subject_id; i++)
		{
			if ( p_scanMultiDBAuxWrap->subject[i] != NULL)
			{
				checkCudaErrors(cudaFree(p_scanMultiDBAuxWrap->subject[i]));
			}
		}
		delete[] p_scanMultiDBAuxWrap->subject;

		delete p_scanMultiDBAuxWrap;   
		gpu_obj->m_global = NULL;
	}
}

//////////////////////////////////////////////////////////////////////////
//small
/** Wrapper structure for different types of cuda structures */
struct cudaSmallTableAuxWrap : public GpuObject
{
	Int2 * overflowtable;
	Uint4 overflowtable_len;
	Uint1 * query_compressed_nuc_seq_start;
	BlastContextInfo* contextinfo;
	Int2 * backbone;
	//Uint4 * d_small_table;
	Uint4 * h_small_table;
} ;

///static cudaSmallTableAuxWrap cuda_smallAuxWrap;
//////////////////////////////////////////////////////////////////////////
//init query gpu memory
void InitSmallNaLookupTableTexture_v3(cudaSmallTableAuxWrap& cuda_smallAuxWrap, LookupTableWrap* lookup_wrap)
{
	BlastSmallNaLookupTable *lookup = (BlastSmallNaLookupTable *) lookup_wrap->lut;
	Int4 backbone_size = lookup->backbone_size;

	Int2* d_backbone = NULL;
	//create cuda
	checkCudaErrors(cudaMalloc( (void**)&d_backbone, backbone_size * sizeof (Int2)));
	checkCudaErrors(cudaMemcpy( d_backbone, lookup->final_backbone, backbone_size * sizeof (Int2), cudaMemcpyHostToDevice));
	SET_INT2_BASE;
	cuda_smallAuxWrap.backbone = d_backbone;

	//////////////////////////////////////////////////////////////////////////
	//
	Uint4 * h_small_tb = new Uint4[backbone_size/32/2]();
	Int2 hash_value1 = -1;
	Int2 hash_value2 = -1;
	int k = 0; 
	for (int i = 0; i < backbone_size/32/2; i++)
	{
		Uint4 bit_hash_value = 0;
		for ( int j = 0; j < 32; j++)
		{
			hash_value1 = lookup->final_backbone[k++];
			hash_value2 = lookup->final_backbone[k++];
			if (hash_value1 != -1 || hash_value2 != -1)
			{
				bit_hash_value |= 1<<j;
			}
		}
		h_small_tb[i] = bit_hash_value;
	}
	cuda_smallAuxWrap.h_small_table = h_small_tb;
	checkCudaErrors(cudaMemcpyToSymbol(cn_small_tb, h_small_tb, 65536/8/2));

}

void InitSmallQueryGPUMem(LookupTableWrap* lookup_wrap, BLAST_SequenceBlk* query, BlastQueryInfo* query_info)
{
	GpuData* gpu_obj = BlastMGPUUtil.GetCurrentThreadGPUData();
	if (gpu_obj->m_local == NULL)
	{
		BlastContextInfo* d_contextinfo = NULL;
		Uint1* d_query_compressed_nuc_seq_start = NULL;
		BlastSmallNaLookupTable *lookup = (BlastSmallNaLookupTable *) lookup_wrap->lut;
		Int2* d_overflow = NULL;

		int context_num = query_info->last_context - query_info->first_context +1;
		checkCudaErrors(cudaMalloc((void **)&d_contextinfo,  context_num * sizeof(BlastContextInfo)));
		checkCudaErrors(cudaMemcpy(d_contextinfo, &query_info->contexts[query_info->first_context], context_num* sizeof(BlastContextInfo), cudaMemcpyHostToDevice));

		int compressed_seq_len = query->length + 3;
		checkCudaErrors(cudaMalloc((void **)&d_query_compressed_nuc_seq_start, (compressed_seq_len) * sizeof(Uint1)));
		checkCudaErrors(cudaMemcpy(d_query_compressed_nuc_seq_start, query->compressed_nuc_seq_start,  (compressed_seq_len) * sizeof(Uint1), cudaMemcpyHostToDevice));

		int overflow_size = lookup->overflow_size;
		checkCudaErrors(cudaMalloc((void**)&d_overflow, overflow_size *(sizeof(Int2))));
		checkCudaErrors(cudaMemcpy( d_overflow, lookup->overflow, overflow_size * sizeof (Int2), cudaMemcpyHostToDevice));


		cudaSmallTableAuxWrap* p_smallAuxWrap = new cudaSmallTableAuxWrap();
		p_smallAuxWrap->contextinfo = d_contextinfo;
		p_smallAuxWrap->query_compressed_nuc_seq_start = d_query_compressed_nuc_seq_start;
		p_smallAuxWrap->overflowtable = d_overflow;
		p_smallAuxWrap->overflowtable_len = overflow_size;

		InitSmallNaLookupTableTexture_v3(*p_smallAuxWrap, lookup_wrap);

		gpu_obj->m_local = p_smallAuxWrap;
	}
}

//////////////////////////////////////////////////////////////////////////

void ReleaseSmallQueryGPUMem()
{
	GpuData* gpu_obj = BlastMGPUUtil.GetCurrentThreadGPUData();
	if (gpu_obj->m_local != NULL)
	{
		cudaSmallTableAuxWrap* p_smallAuxWrap = (cudaSmallTableAuxWrap*) gpu_obj->m_local;
		
		checkCudaErrors(cudaFree(p_smallAuxWrap->contextinfo));
		checkCudaErrors(cudaFree(p_smallAuxWrap->query_compressed_nuc_seq_start));
		checkCudaErrors(cudaFree(p_smallAuxWrap->overflowtable));
		checkCudaErrors(cudaFree(p_smallAuxWrap->backbone));
		checkCudaErrors(cudaUnbindTexture(tx_backbone));
		delete [] p_smallAuxWrap->h_small_table;
		delete p_smallAuxWrap;		
		gpu_obj->m_local = NULL;
	}
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
int s_gpu_MBScanSubject_8_1Mod4_scankernel_Opt_v3_1(Uint4 scan_range_temp,
	cudaScanAuxWrapMultiQueries* p_scanMultiDBAuxWrap,
	cudaSmallTableAuxWrap* p_smallAuxWrap,
	const LookupTableWrap* lookup_wrap,
	const BLAST_SequenceBlk* subject,
	BlastOffsetPair* NCBI_RESTRICT offset_pairs, Int4 max_hits,  
	Int4* scan_range)
{
	Uint4 total_hits = 0;
	int current_subject_id = subject->oid;

	checkCudaErrors(cudaMemset(p_scanMultiDBAuxWrap->total_hits, 0, sizeof(unsigned int)));  //初始化为0
	checkCudaErrors(cudaMemset(p_scanMultiDBAuxWrap->over_hits_num, 0, sizeof(unsigned int)));  //初始化为0

	static int blocksize_x = SHARE_MEM_SMALL_SIZE/2;
	dim3 blockSize(blocksize_x);
	dim3 gridSize;

	scan_range_temp = scan_range_temp/16;
	gridSize.x = (scan_range_temp+ blockSize.x -1)/ blockSize.x;

	static int max_grid_size = 8192;
	if (gridSize.x > max_grid_size)
	{
		gridSize.x = max_grid_size;
	}

	Uint4 global_size = gridSize.x;
	global_size *= blocksize_x;

	slogfile.KernelStart();

	gpu_blastn_scan_8_1mod4_v3<<< gridSize, blockSize >>>(
		p_scanMultiDBAuxWrap->subject[current_subject_id],
		p_scanMultiDBAuxWrap->offsetPairs,
		p_scanMultiDBAuxWrap->over_offset_pairs,
		p_scanMultiDBAuxWrap->total_hits,
		p_scanMultiDBAuxWrap->over_hits_num,
		scan_range_temp, 
		scan_range[0],
		global_size,
		p_smallAuxWrap->backbone);


	getLastCudaError("gpu_blastn_scan_8_1mod4() execution failed.\n");

	slogfile.KernelEnd();
	slogfile.addTotalTime("scan_kernel_time", slogfile.KernelElaplsedTime(), false);

	checkCudaErrors(cudaMemcpy(&total_hits, p_scanMultiDBAuxWrap->over_hits_num, sizeof(Uint4), cudaMemcpyDeviceToHost));
	
	return total_hits;
}

int s_gpu_MBScanSubject_8_1Mod4_scankernel_Opt_v3_2(int total_hits,
	cudaScanAuxWrapMultiQueries* p_scanMultiDBAuxWrap,
	cudaSmallTableAuxWrap* p_smallAuxWrap,
	const LookupTableWrap* lookup_wrap,
	const BLAST_SequenceBlk* subject,
	BlastOffsetPair* NCBI_RESTRICT offset_pairs, Int4 max_hits,  
	Int4* scan_range)
{	
	if (total_hits > 0)
	{
		Int4 threadNum = 256;
		Int4 blockNum = (total_hits + threadNum - 1)/threadNum;
		dim3 gridDim(blockNum, 1);
		dim3 blockDim(threadNum, 1);

		//checkCudaErrors(cudaMemset(cuda_extenWrap.exact_hits_num, 0, sizeof(unsigned int)));  //初始化为0
#if LOG_TIME
		slogfile.KernelStart();
#endif

		kernel_lookupInSmallTable_v3<<<gridDim,blockDim>>>(
			p_smallAuxWrap->overflowtable,
			total_hits,
			p_scanMultiDBAuxWrap->over_offset_pairs,
			p_scanMultiDBAuxWrap->offsetPairs,
			p_scanMultiDBAuxWrap->total_hits,
			p_smallAuxWrap->overflowtable_len
			);
		getLastCudaError("kernel_lookupInsSmallTable() execution failed.\n");
#if LOG_TIME
		slogfile.KernelEnd();
		slogfile.addTotalTime("lookup_kernel_time", slogfile.KernelElaplsedTime() ,false );
#endif
	}
		
	checkCudaErrors(cudaMemcpy(&total_hits, p_scanMultiDBAuxWrap->total_hits, sizeof(unsigned int), cudaMemcpyDeviceToHost)); 
#if LOG_TIME
	slogfile.addTotalNum("Kernel_lookupInBigHashTable hits", total_hits,false);
#endif

	return total_hits;
}
Int4 
	s_gpu_MBScanSubject_8_1Mod4_scankernel_Opt_v3(const LookupTableWrap* lookup_wrap,
	const BLAST_SequenceBlk* subject,
	BlastOffsetPair* NCBI_RESTRICT offset_pairs, Int4 max_hits,  
	Int4* scan_range)
{
	BlastSmallNaLookupTable *lookup = (BlastSmallNaLookupTable *) lookup_wrap->lut;
	Uint4 total_hits = 0; 

	max_hits -= lookup->longest_chain;
	ASSERT(lookup_wrap->lut_type == eSmallNaLookupTable);
	ASSERT(lookup->lut_word_length == 8);
	ASSERT(lookup->scan_step % COMPRESSION_RATIO == 1);

	if(scan_range[0] > scan_range[1]) return 0;

	max_hits -= lookup->longest_chain;

	Uint4 scan_range_temp = (scan_range[1]+lookup->lut_word_length - scan_range[0]);

	int current_subject_id = subject->oid;

	GpuData* gpu_obj = BlastMGPUUtil.GetCurrentThreadGPUData();
	cudaScanAuxWrapMultiQueries* p_scanMultiDBAuxWrap = (cudaScanAuxWrapMultiQueries*) gpu_obj->m_global;
	cudaSmallTableAuxWrap* p_smallAuxWrap = (cudaSmallTableAuxWrap*) gpu_obj->m_local;

	slogfile.Start();
	if (p_scanMultiDBAuxWrap->subject[current_subject_id] == NULL)
	{	
		//printf("%d,\n", current_subject_id);
		p_scanMultiDBAuxWrap->subject_id = current_subject_id;
		checkCudaErrors(cudaMalloc((void **)&p_scanMultiDBAuxWrap->subject[current_subject_id],(scan_range_temp)/4));
		checkCudaErrors(cudaMemcpy(p_scanMultiDBAuxWrap->subject[current_subject_id], subject->sequence, (scan_range_temp) / 4 , cudaMemcpyHostToDevice));
	}
	slogfile.End();
	slogfile.addTotalTime("Scan CPU -> GPU Memory Time",slogfile.elaplsedTime(), false);

	total_hits = s_gpu_MBScanSubject_8_1Mod4_scankernel_Opt_v3_1(scan_range_temp,p_scanMultiDBAuxWrap, p_smallAuxWrap,lookup_wrap,subject,offset_pairs,max_hits,scan_range);
	total_hits = s_gpu_MBScanSubject_8_1Mod4_scankernel_Opt_v3_2(total_hits,p_scanMultiDBAuxWrap, p_smallAuxWrap, lookup_wrap,subject,offset_pairs,max_hits,scan_range);

	scan_range[0] = scan_range[1]+lookup->lut_word_length;

	return total_hits;
}

Int4 
	s_gpu_BlastSmallExtend_v3(BlastOffsetPair * offset_pairs, Int4 num_hits,
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
	BlastSmallNaLookupTable *lut = (BlastSmallNaLookupTable *) lookup_wrap->lut;
	word_length = lut->word_length;
	lut_word_length = lut->lut_word_length;

	int current_subject_id = subject->oid;
	
	unsigned int exact_hits_num = 0;

	Int4 threadNum = 512;
	Int4 blockNum_Ex = (num_hits + threadNum - 1)/threadNum;
	dim3 gridDim_Ex(blockNum_Ex, 1);
	dim3 blockDim_Ex(threadNum, 1);

	GpuData* gpu_obj = BlastMGPUUtil.GetCurrentThreadGPUData();
	cudaScanAuxWrapMultiQueries* p_scanMultiDBAuxWrap = (cudaScanAuxWrapMultiQueries*) gpu_obj->m_global;
	cudaSmallTableAuxWrap* p_smallAuxWrap = (cudaSmallTableAuxWrap*) gpu_obj->m_local;

	checkCudaErrors(cudaMemset(p_scanMultiDBAuxWrap->over_hits_num, 0, sizeof(unsigned int)));  //初始化为0

	//printf("id:%d,num_his:%d \n", p_scanMultiDBAuxWrap->subject_id,num_hits);
	slogfile.KernelStart();

	kernel_s_BlastSmallExtend_v3<<<gridDim_Ex, blockDim_Ex>>> (
		(Uint1*)p_scanMultiDBAuxWrap->subject[current_subject_id],
		p_smallAuxWrap->query_compressed_nuc_seq_start,
		word_length,
		lut_word_length,	
		num_hits,
		p_scanMultiDBAuxWrap->offsetPairs,p_scanMultiDBAuxWrap->over_offset_pairs,  
		p_scanMultiDBAuxWrap->over_hits_num, 
		s_range,
		p_smallAuxWrap->contextinfo,
		query_info->last_context);
	getLastCudaError("kernel_s_BlastSmallExtend() execution failed.\n");

	slogfile.KernelEnd();
	slogfile.addTotalTime("extend_kernel_time", slogfile.KernelElaplsedTime(), false);

	checkCudaErrors(cudaMemcpy(&exact_hits_num, p_scanMultiDBAuxWrap->over_hits_num, sizeof(unsigned int), cudaMemcpyDeviceToHost));
	slogfile.addTotalNum("Small_extended hits", exact_hits_num,false);

	__int64 c1 = 0,c2 = 0;
	if (exact_hits_num >0)
	{
		c1 = slogfile.Start();
		thrust::device_ptr<BlastOffsetPair> dev_ptr(p_scanMultiDBAuxWrap->over_offset_pairs);
		thrust::sort(dev_ptr,dev_ptr+exact_hits_num,OffsetPairCmp());
		checkCudaErrors(cudaMemcpy(offset_pairs, p_scanMultiDBAuxWrap->over_offset_pairs, exact_hits_num * sizeof(BlastOffsetPair), cudaMemcpyDeviceToHost));
		c2 = slogfile.End();
		slogfile.addTotalTime("GPU->CPU memory Time", c1, c2, false);
		//thrust::sort(offset_pairs,offset_pairs+exact_hits_num,OffsetPairCmp());
	}

	c1 = slogfile.Start();
	if (exact_hits_num >0)
	{  
		if (word_params->container_type == eDiagHash) {
			for(int i=0; i<exact_hits_num; i++){
				Int4 s_offset = offset_pairs[i].qs_offsets.s_off;
				Int4 q_offset = offset_pairs[i].qs_offsets.q_off;
			hits_extended += s_BlastnDiagHashExtendInitialHit(query, subject, 
				q_offset, s_offset,  
				lut->masked_locations, 
				query_info, s_range, 
				word_length, lut_word_length,
				lookup_wrap,
				word_params, matrix,
				ewp->hash_table,
				init_hitlist);
			}
		}
		else{
			for(int i=0; i<exact_hits_num; i++){
				Int4 s_offset = offset_pairs[i].qs_offsets.s_off;
				Int4 q_offset = offset_pairs[i].qs_offsets.q_off;	
				hits_extended += s_BlastnDiagTableExtendInitialHit(query, subject, 
					q_offset, s_offset,  
					lut->masked_locations, 
					query_info, s_range, 
					word_length, lut_word_length,
					lookup_wrap,
					word_params, matrix,
					ewp->diag_table,
					init_hitlist);
			}
		}


	}


	c2 = slogfile.End();
	slogfile.addTotalTime("Hits extend time",c1,c2, false);

	return hits_extended; 
}

//////////////////////////////////////////////////////////////////////////
///blastn
//////////////////////////////////////////////////////////////////////////

int s_gpu_BlastSmallNaScanSubject_8_4_p1(Uint4 scan_range_temp,
	cudaScanAuxWrapMultiQueries* p_scanMultiDBAuxWrap,
	cudaSmallTableAuxWrap* p_smallAuxWrap,
	const LookupTableWrap* lookup_wrap,
	const BLAST_SequenceBlk* subject,
	BlastOffsetPair* NCBI_RESTRICT offset_pairs, Int4 max_hits,  
	Int4* scan_range)
{
	Uint4 total_hits = 0;
	int current_subject_id = subject->oid;

	//checkCudaErrors(cudaMemset(p_scanMultiDBAuxWrap->total_hits, 0, sizeof(unsigned int)));  //初始化为0
	//checkCudaErrors(cudaMemset(p_scanMultiDBAuxWrap->over_hits_num, 0, sizeof(unsigned int)));  //初始化为0
	checkCudaErrors(cudaMemcpy(p_scanMultiDBAuxWrap->counter, p_scanMultiDBAuxWrap->zero, 3*sizeof(Uint4), cudaMemcpyDeviceToDevice));
	//cout << subject->length << "\t";

#if 0
	//static int blocksize_x = SHARE_MEM_SMALL_SIZE/2;
	static int blocksize_x = 256;
	dim3 blockSize(blocksize_x);
	dim3 gridSize;

	scan_range_temp = scan_range_temp/32;
	gridSize.x = (scan_range_temp+ blockSize.x -1)/ blockSize.x;

	static int max_grid_size = 16384;
	if (gridSize.x > max_grid_size)
	{
		gridSize.x = max_grid_size;
	}				  

	Uint4 global_size = gridSize.x;
	global_size *= blocksize_x;

#if LOG_TIME
	slogfile.KernelStart();
#endif

	gpu_blastn_scan_8_4<<< gridSize, blockSize>>>(
		(Uint8*)p_scanMultiDBAuxWrap->subject[current_subject_id],
		p_scanMultiDBAuxWrap->offsetPairs,
		p_scanMultiDBAuxWrap->over_offset_pairs,
		&p_scanMultiDBAuxWrap->counter[0],
		&p_scanMultiDBAuxWrap->counter[1],
		scan_range_temp, 
		scan_range[0],
		global_size,
		p_smallAuxWrap->backbone);
#endif

#if 0
static int blocksize_x = SHARE_MEM_SMALL_SIZE/2;
	dim3 blockSize(blocksize_x);
	dim3 gridSize;

	scan_range_temp = scan_range_temp/4;
	gridSize.x = (scan_range_temp+ blockSize.x -1)/ blockSize.x;

	static int max_grid_size = 4096;
	if (gridSize.x > max_grid_size)
	{
		gridSize.x = max_grid_size;
	}				  

	Uint4 global_size = gridSize.x;
	global_size *= blocksize_x;

#if LOG_TIME
	slogfile.KernelStart();
#endif

	gpu_blastn_scan_8_4_v1<<< gridSize, blockSize>>>(
		(Uint1*)p_scanMultiDBAuxWrap->subject[current_subject_id],
		p_scanMultiDBAuxWrap->offsetPairs,
		p_scanMultiDBAuxWrap->over_offset_pairs,
		&p_scanMultiDBAuxWrap->counter[0],
		&p_scanMultiDBAuxWrap->counter[1],
		scan_range_temp, 
		scan_range[0],
		global_size,
		p_smallAuxWrap->backbone);
#endif

#if 0
	static int blocksize_x = 256;
	dim3 blockSize(blocksize_x);
	dim3 gridSize;

	scan_range_temp = (scan_range_temp-15)/16;
	gridSize.x = (scan_range_temp+ blockSize.x -1)/ blockSize.x;

	static int max_grid_size = 16384;
	if (gridSize.x > max_grid_size)
	{
		gridSize.x = max_grid_size;
	}				  
	Uint4 global_size = gridSize.x;
	global_size *= blocksize_x;		   

#if LOG_TIME
	slogfile.KernelStart();
#endif

	gpu_blastn_scan_8_4_v2<<< gridSize, blockSize>>>(
		p_scanMultiDBAuxWrap->subject[current_subject_id],
		p_scanMultiDBAuxWrap->offsetPairs,
		p_scanMultiDBAuxWrap->over_offset_pairs,
		&p_scanMultiDBAuxWrap->counter[0],
		&p_scanMultiDBAuxWrap->counter[1],
		scan_range_temp, 
		scan_range[0],
		global_size,
		p_smallAuxWrap->backbone);
#endif
#if 0  
	static int blocksize_x = 256 ;
	dim3 blockSize(blocksize_x);
	dim3 gridSize;

	scan_range_temp = (scan_range_temp-15)/16;
	gridSize.x = (scan_range_temp+ blockSize.x -1)/ blockSize.x;

	static int max_grid_size = 256;
	if (gridSize.x > max_grid_size)
	{
		gridSize.x = max_grid_size;
	}				  
	Uint4 global_size = gridSize.x;
	global_size *= blocksize_x;		   

#if LOG_TIME
	slogfile.KernelStart();
#endif		  
	gpu_blastn_scan_8_4_v2_1<<< gridSize, blockSize>>>(
		p_scanMultiDBAuxWrap->subject[current_subject_id],
		p_scanMultiDBAuxWrap->offsetPairs,
		p_scanMultiDBAuxWrap->over_offset_pairs,
		&p_scanMultiDBAuxWrap->counter[0],
		&p_scanMultiDBAuxWrap->counter[1],
		scan_range_temp, 
		scan_range[0],
		global_size,
		p_smallAuxWrap->backbone);
#endif
#if 0
	//static int blocksize_x = SHARE_MEM_SMALL_SIZE/2;
	static int blocksize_x = 512;
	dim3 blockSize(blocksize_x);
	dim3 gridSize;

	//scan_range_temp = scan_range_temp/32;
	scan_range_temp = scan_range_temp/4;
	gridSize.x = (scan_range_temp+ blockSize.x -1)/ blockSize.x;

	static int max_grid_size = 16384;
	if (gridSize.x > max_grid_size)
	{
		gridSize.x = max_grid_size;
	}				  

	Uint4 global_size = gridSize.x;
	global_size *= blocksize_x;

#if LOG_TIME
	slogfile.KernelStart();
#endif

	gpu_blastn_scan_8_4_v4<<< gridSize, blockSize>>>(
		(Uint1*)p_scanMultiDBAuxWrap->subject[current_subject_id],
		p_scanMultiDBAuxWrap->offsetPairs,
		p_scanMultiDBAuxWrap->over_offset_pairs,
		p_scanMultiDBAuxWrap->counter,
		scan_range_temp, 
		scan_range[0],
		global_size,
		p_smallAuxWrap->backbone);
#endif
#if 0
	static int blocksize_x = 256;
	dim3 blockSize(blocksize_x);
	dim3 gridSize;

	scan_range_temp = (scan_range_temp-15)/16;
	gridSize.x = (scan_range_temp+ blockSize.x -1)/ blockSize.x;

	static int max_grid_size = 16384;
	if (gridSize.x > max_grid_size)
	{
		gridSize.x = max_grid_size;
	}				  
	Uint4 global_size = gridSize.x;
	global_size *= blocksize_x;		   

#if LOG_TIME
	slogfile.KernelStart();
#endif

	gpu_blastn_scan_8_4_v3<<< gridSize, blockSize>>>(
		p_scanMultiDBAuxWrap->subject[current_subject_id],
		p_scanMultiDBAuxWrap->offsetPairs,
		p_scanMultiDBAuxWrap->over_offset_pairs,
		&p_scanMultiDBAuxWrap->counter[0],
		&p_scanMultiDBAuxWrap->counter[1],
		scan_range_temp, 
		scan_range[0],
		global_size,
		p_smallAuxWrap->backbone);
#endif

#if 0
	static int blocksize_x = SHARE_MEM_SMALL_SIZE_V5/2;
	dim3 blockSize(blocksize_x);
	dim3 gridSize;

	scan_range_temp = scan_range_temp/32;
	gridSize.x = (scan_range_temp+ blockSize.x -1)/ blockSize.x;

	static int max_grid_size = 16384;
	if (gridSize.x > max_grid_size)
	{
		gridSize.x = max_grid_size;
	}				  

	Uint4 global_size = gridSize.x;
	global_size *= blocksize_x;

#if LOG_TIME
	slogfile.KernelStart();
#endif			
	gpu_blastn_scan_8_4_v5<<< gridSize, blockSize>>>(
		(Uint8*)p_scanMultiDBAuxWrap->subject[current_subject_id],
		p_scanMultiDBAuxWrap->offsetPairs,
		p_scanMultiDBAuxWrap->over_offset_pairs,
		&p_scanMultiDBAuxWrap->counter[0],
		&p_scanMultiDBAuxWrap->counter[1],
		scan_range_temp, 
		scan_range[0],
		global_size,
		p_smallAuxWrap->backbone);
#endif

#if 1
	static int blocksize_x = 512;
	dim3 blockSize(blocksize_x);
	dim3 gridSize;
	scan_range_temp = (scan_range_temp-15)/16;
	gridSize.x = (scan_range_temp+ blockSize.x -1)/ blockSize.x;
	static int max_grid_size = 128;
	if (gridSize.x > max_grid_size)
	{
		gridSize.x = max_grid_size;
	}				  
	Uint4 global_size = gridSize.x;
	global_size *= blocksize_x;

#if LOG_TIME
	slogfile.KernelStart();
#endif		  
	gpu_blastn_scan_8_4_v2_2<<< gridSize, blockSize>>>(
		p_scanMultiDBAuxWrap->subject[current_subject_id],
		p_scanMultiDBAuxWrap->offsetPairs,
		p_scanMultiDBAuxWrap->over_offset_pairs,
		&p_scanMultiDBAuxWrap->counter[0],
		&p_scanMultiDBAuxWrap->counter[1],
		scan_range_temp, 
		scan_range[0],
		global_size,
		p_smallAuxWrap->backbone);
#endif		 
	getLastCudaError("gpu_blastn_scan_8_4_v1() execution failed.\n");
#if LOG_TIME
	slogfile.KernelEnd();
	slogfile.addTotalTime("scan_kernel_time", slogfile.KernelElaplsedTime(), false);
#endif			   
	//checkCudaErrors(cudaMemcpy(&total_hits, &p_scanMultiDBAuxWrap->counter[0], sizeof(Uint4), cudaMemcpyDeviceToHost));
	//cout << total_hits <<"\t";
	checkCudaErrors(cudaMemcpy(&total_hits, &p_scanMultiDBAuxWrap->counter[1], sizeof(Uint4), cudaMemcpyDeviceToHost));
	//cout << total_hits <<"\n";	

	return total_hits;
}

int s_gpu_BlastSmallNaScanSubject_8_4_p2(int total_hits,
	cudaScanAuxWrapMultiQueries* p_scanMultiDBAuxWrap,
	cudaSmallTableAuxWrap* p_smallAuxWrap,
	const LookupTableWrap* lookup_wrap,
	const BLAST_SequenceBlk* subject,
	BlastOffsetPair* NCBI_RESTRICT offset_pairs, Int4 max_hits,  
	Int4* scan_range)
{	
	if (total_hits > 0)
	{
		Int4 threadNum = 256;
		Int4 blockNum = (total_hits + threadNum - 1)/threadNum;
		dim3 gridDim(blockNum, 1);
		dim3 blockDim(threadNum, 1);

		//checkCudaErrors(cudaMemset(cuda_extenWrap.exact_hits_num, 0, sizeof(unsigned int)));  //初始化为0
#if LOG_TIME
		slogfile.KernelStart();
#endif

		//kernel_lookupInSmallTable_v3<<<gridDim,blockDim>>>(
		kernel_lookupInSmallTable_v3<<<gridDim,blockDim>>>(
			p_smallAuxWrap->overflowtable,
			total_hits,
			p_scanMultiDBAuxWrap->over_offset_pairs,
			p_scanMultiDBAuxWrap->offsetPairs,
			&p_scanMultiDBAuxWrap->counter[0],
			p_smallAuxWrap->overflowtable_len
			);
		getLastCudaError("kernel_lookupInsSmallTable() execution failed.\n");
#if LOG_TIME
		slogfile.KernelEnd();
		slogfile.addTotalTime("lookup_kernel_time", slogfile.KernelElaplsedTime() ,false );
#endif
	}

	checkCudaErrors(cudaMemcpy(&total_hits, &p_scanMultiDBAuxWrap->counter[0], sizeof(unsigned int), cudaMemcpyDeviceToHost)); 
#if LOG_TIME
	slogfile.addTotalNum("Kernel_lookupInBigHashTable hits", total_hits,false);
#endif

	return total_hits;
}

Int4 
	s_gpu_BlastSmallNaScanSubject_8_4(const LookupTableWrap* lookup_wrap,
	const BLAST_SequenceBlk* subject,
	BlastOffsetPair* NCBI_RESTRICT offset_pairs, Int4 max_hits,  
	Int4* scan_range)
{
	BlastSmallNaLookupTable *lookup = (BlastSmallNaLookupTable *) lookup_wrap->lut;
	Uint4 total_hits = 0; 

	max_hits -= lookup->longest_chain;
	ASSERT(lookup_wrap->lut_type == eSmallNaLookupTable);
	ASSERT(lookup->lut_word_length == 8);
	//ASSERT(lookup->scan_step % COMPRESSION_RATIO == 1);

	if(scan_range[0] > scan_range[1]) return 0;

	max_hits -= lookup->longest_chain;

	Uint4 scan_range_temp = (scan_range[1]+lookup->lut_word_length - scan_range[0]);

	int current_subject_id = subject->oid;

	GpuData* gpu_obj = BlastMGPUUtil.GetCurrentThreadGPUData();
	cudaScanAuxWrapMultiQueries* p_scanMultiDBAuxWrap = (cudaScanAuxWrapMultiQueries*) gpu_obj->m_global;
	cudaSmallTableAuxWrap* p_smallAuxWrap = (cudaSmallTableAuxWrap*) gpu_obj->m_local;
#if LOG_TIME
	  slogfile.Start();
#endif
	if (p_scanMultiDBAuxWrap->subject[current_subject_id] == NULL)
	{	
		//printf("%d,\n", current_subject_id);
		p_scanMultiDBAuxWrap->subject_id = current_subject_id;
		checkCudaErrors(cudaMalloc((void **)&p_scanMultiDBAuxWrap->subject[current_subject_id],(scan_range_temp)/4));
		checkCudaErrors(cudaMemcpy(p_scanMultiDBAuxWrap->subject[current_subject_id], subject->sequence, (scan_range_temp) / 4 , cudaMemcpyHostToDevice));
	}
#if LOG_TIME
	slogfile.End();
	slogfile.addTotalTime("Scan CPU -> GPU Memory Time",slogfile.elaplsedTime(), false);
#endif
	total_hits = s_gpu_BlastSmallNaScanSubject_8_4_p1(scan_range_temp,p_scanMultiDBAuxWrap, p_smallAuxWrap,lookup_wrap,subject,offset_pairs,max_hits,scan_range);
	total_hits = s_gpu_BlastSmallNaScanSubject_8_4_p2(total_hits,p_scanMultiDBAuxWrap, p_smallAuxWrap, lookup_wrap,subject,offset_pairs,max_hits,scan_range);

	scan_range[0] = scan_range[1]+lookup->lut_word_length;

	return total_hits;
}

Int4 
	s_gpu_BlastSmallNaExtendAlignedOneByte(BlastOffsetPair * offset_pairs, Int4 num_hits,
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
	BlastSmallNaLookupTable *lut = (BlastSmallNaLookupTable *) lookup_wrap->lut;
	word_length = lut->word_length;
	lut_word_length = lut->lut_word_length;

	int current_subject_id = subject->oid;

	unsigned int exact_hits_num = 0;

	Int4 threadNum = 512;
	Int4 blockNum_Ex = (num_hits + threadNum - 1)/threadNum;
	dim3 gridDim_Ex(blockNum_Ex, 1);
	dim3 blockDim_Ex(threadNum, 1);

	GpuData* gpu_obj = BlastMGPUUtil.GetCurrentThreadGPUData();
	cudaScanAuxWrapMultiQueries* p_scanMultiDBAuxWrap = (cudaScanAuxWrapMultiQueries*) gpu_obj->m_global;
	cudaSmallTableAuxWrap* p_smallAuxWrap = (cudaSmallTableAuxWrap*) gpu_obj->m_local;

	//checkCudaErrors(cudaMemset(p_scanMultiDBAuxWrap->over_hits_num, 0, sizeof(unsigned int)));  //初始化为0

	//printf("id:%d,num_his:%d \n", p_scanMultiDBAuxWrap->subject_id,num_hits);
#if LOG_TIME
	slogfile.KernelStart();
#endif

	kernel_s_BlastSmallNaExtendAlignedOneByte<<<gridDim_Ex, blockDim_Ex>>> (
		(Uint1*)p_scanMultiDBAuxWrap->subject[current_subject_id],
		p_smallAuxWrap->query_compressed_nuc_seq_start,
		word_length,
		lut_word_length,	
		num_hits,
		p_scanMultiDBAuxWrap->offsetPairs,p_scanMultiDBAuxWrap->over_offset_pairs,  
		&p_scanMultiDBAuxWrap->counter[2], 
		s_range,
		p_smallAuxWrap->contextinfo,
		query_info->last_context,
		query->length);
	getLastCudaError("kernel_s_BlastSmallExtend() execution failed.\n");
#if LOG_TIME
	slogfile.KernelEnd();
	slogfile.addTotalTime("extend_kernel_time", slogfile.KernelElaplsedTime(), false);
#endif
	checkCudaErrors(cudaMemcpy(&exact_hits_num, &p_scanMultiDBAuxWrap->counter[2], sizeof(Uint4), cudaMemcpyDeviceToHost));
	slogfile.addTotalNum("Small_extended hits", exact_hits_num,false);

	__int64 c1 = 0,c2 = 0;
	if (exact_hits_num >0)
	{
#if LOG_TIME
		c1 = slogfile.Start();
#endif
		//thrust::device_ptr<BlastOffsetPair> dev_ptr(p_scanMultiDBAuxWrap->over_offset_pairs);
		//thrust::sort(dev_ptr,dev_ptr+exact_hits_num,OffsetPairCmp());
		checkCudaErrors(cudaMemcpy(offset_pairs, p_scanMultiDBAuxWrap->over_offset_pairs, exact_hits_num * sizeof(BlastOffsetPair), cudaMemcpyDeviceToHost));
#if LOG_TIME		
		c2 = slogfile.End();
		slogfile.addTotalTime("GPU->CPU memory Time", c1, c2, false);
#endif
		
		thrust::sort(offset_pairs,offset_pairs+exact_hits_num,OffsetPairCmp());
	}
#if LOG_TIME
	c1 = slogfile.Start();
#endif
	if (exact_hits_num >0)
	{  
		if (word_params->container_type == eDiagHash) {
			for(int i=0; i<exact_hits_num; i++){
				Int4 s_offset = offset_pairs[i].qs_offsets.s_off;
				Int4 q_offset = offset_pairs[i].qs_offsets.q_off;
				hits_extended += s_BlastnDiagHashExtendInitialHit(query, subject, 
					q_offset, s_offset,  
					lut->masked_locations, 
					query_info, s_range, 
					word_length, lut_word_length,
					lookup_wrap,
					word_params, matrix,
					ewp->hash_table,
					init_hitlist);
			}
		}
		else{
			//#pragma omp parallel for
			for(int i=0; i<exact_hits_num; i++){
				Int4 s_offset = offset_pairs[i].qs_offsets.s_off;
				Int4 q_offset = offset_pairs[i].qs_offsets.q_off;	
				hits_extended += s_BlastnDiagTableExtendInitialHit(query, subject, 
					q_offset, s_offset,  
					lut->masked_locations, 
					query_info, s_range, 
					word_length, lut_word_length,
					lookup_wrap,
					word_params, matrix,
					ewp->diag_table,
					init_hitlist);
			}
		}


	}

#if LOG_TIME
	c2 = slogfile.End();
	slogfile.addTotalTime("Hits extend time",c1,c2, false);
#endif
	return hits_extended; 
}

//////////////////////////////////////////////////////////////////////////
/** Wrapper structure for different types of cuda structures */
struct cudaMBHashAuxWrap : public GpuObject
{
	PV_ARRAY_TYPE* lookupArray;
	Int4 * hashtable;
	Int4 * next_pos;
	Int4 next_pos_len;
	Uint1 * query;
} ;

//static cudaMBHashAuxWrap p_MBHashWrap->;

//////////////////////////////////////////////////////////////////////////
//extend
//////////////////////////////////////////////////////////////////////////
void InitMBLookupTableTexture(cudaMBHashAuxWrap& p_MBHashWrap, LookupTableWrap* lookup_wrap)
{
	BlastMBLookupTable* mb_lt = (BlastMBLookupTable*) lookup_wrap->lut;
	Int4 pv_size = mb_lt->hashsize/(1<<mb_lt->pv_array_bts);

	PV_ARRAY_TYPE* d_lookupArray = NULL;
	//create cuda array
	checkCudaErrors(cudaMalloc( (void**) &d_lookupArray, pv_size *sizeof(PV_ARRAY_TYPE)));
	checkCudaErrors(cudaMemcpy( d_lookupArray, mb_lt->pv_array, pv_size * sizeof (PV_ARRAY_TYPE), cudaMemcpyHostToDevice));
	// Bind the array to the texture
	SET_PVARRAY_BASE;

	p_MBHashWrap.lookupArray = d_lookupArray;
}

void InitMBQueryGPUMem(LookupTableWrap * lookup_wrap,	BLAST_SequenceBlk * query)
{
	GpuData* gpu_obj = BlastMGPUUtil.GetCurrentThreadGPUData();
	if (gpu_obj->m_local == NULL)
	{
		//unsigned int * d_exact_hits_num;
		Int4 * d_hashtable, *d_next_pos;
		Uint1 * d_query;
		//BlastOffsetPair *d_exact_offset_pairs;
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

		cudaMBHashAuxWrap* p_MBHashWrap = new cudaMBHashAuxWrap();
		p_MBHashWrap->hashtable = d_hashtable;
		p_MBHashWrap->next_pos = d_next_pos;
		p_MBHashWrap->query = d_query;
		p_MBHashWrap->next_pos_len = biggest_length_next;

		InitMBLookupTableTexture(*p_MBHashWrap, lookup_wrap);

		gpu_obj->m_local = p_MBHashWrap;
	}
}

void ReleaseMBQueryGPUMem()
{
	GpuData* gpu_obj = BlastMGPUUtil.GetCurrentThreadGPUData();
	if (gpu_obj->m_local != NULL)
	{
		cudaMBHashAuxWrap* p_MBHashWrap = (cudaMBHashAuxWrap*) gpu_obj->m_local;

		checkCudaErrors(cudaFree(p_MBHashWrap->hashtable));
		checkCudaErrors(cudaFree(p_MBHashWrap->next_pos));
		checkCudaErrors(cudaFree(p_MBHashWrap->query));
		checkCudaErrors(cudaFree(p_MBHashWrap->lookupArray));
		checkCudaErrors(cudaUnbindTexture(tx_pv_array));

		delete p_MBHashWrap;		
		gpu_obj->m_local = NULL;
	}
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
Int4 
	s_gpu_MBScanSubject_11_2Mod4_scankernel_Opt_v3_1(Uint4 scan_range_temp,
	cudaScanAuxWrapMultiQueries* p_scanMultiDBAuxWrap,
	cudaMBHashAuxWrap* p_MBHashWrap,
	const LookupTableWrap* lookup_wrap,
	const BLAST_SequenceBlk* subject,
	BlastOffsetPair* NCBI_RESTRICT offset_pairs, Int4 max_hits,  
	Int4* scan_range)
{

	BlastMBLookupTable* mb_lt = (BlastMBLookupTable*) lookup_wrap->lut;
	Uint4 total_hits = 0;
	Int4 top_shift =2; 
	Int4 pv_array_bts = mb_lt->pv_array_bts;

	int current_subject_id = subject->oid;

	checkCudaErrors(cudaMemset(p_scanMultiDBAuxWrap->total_hits, 0, sizeof(unsigned int)));  //初始化为0

	static int blocksize_x = SHARE_MEM_SIZE/2;
	dim3 blockSize(blocksize_x);
	dim3 gridSize;
	//scan_range_temp = (scan_range_temp/16) + (scan_range_temp%16);
	scan_range_temp = scan_range_temp/16;
	gridSize.x = (scan_range_temp+ blockSize.x -1)/ blockSize.x;

	static int max_grid_size = 16384;
	if (gridSize.x > max_grid_size)
	{
		gridSize.x = max_grid_size;
	}

	Uint4 global_size = gridSize.x;
	global_size *= blocksize_x;

	slogfile.KernelStart();
	//printf("%d %d\n", gridSize.x, blockSize.x);

	gpu_blastn_scan_11_2mod4_v3<<< gridSize, blockSize >>>(
		p_scanMultiDBAuxWrap->subject[current_subject_id], 
		p_scanMultiDBAuxWrap->offsetPairs, 
		p_scanMultiDBAuxWrap->total_hits, 
		scan_range_temp, 
		scan_range[0], 
		top_shift, 
		pv_array_bts,
		global_size,
		p_MBHashWrap->lookupArray); 

	getLastCudaError("gpu_blastn_scan_11_2mod4_v3() execution failed.\n");

	slogfile.KernelEnd();
	slogfile.addTotalTime("scan_kernel_time", slogfile.KernelElaplsedTime(),false);

	checkCudaErrors(cudaMemcpy(&total_hits, p_scanMultiDBAuxWrap->total_hits, sizeof(Uint4), cudaMemcpyDeviceToHost));

	return total_hits;
}

Int4 
	s_gpu_MBScanSubject_11_2Mod4_scankernel_Opt_v3_2(int total_hits,
	cudaScanAuxWrapMultiQueries* p_scanMultiDBAuxWrap,
	cudaMBHashAuxWrap* p_MBHashWrap,
	const LookupTableWrap* lookup_wrap,
	const BLAST_SequenceBlk* subject,
	BlastOffsetPair* NCBI_RESTRICT offset_pairs, Int4 max_hits,  
	Int4* scan_range)
{
	if (total_hits > 0)
	{
		Int4 threadNum = 512;
		Int4 blockNum = (total_hits + threadNum - 1)/threadNum;
		dim3 gridDim(blockNum, 1);
		dim3 blockDim(threadNum, 1);

		checkCudaErrors(cudaMemset(p_scanMultiDBAuxWrap->over_hits_num, 0, sizeof(unsigned int)));  //初始化为0

		slogfile.KernelStart();
		//cout << total_hits << endl; 

		kernel_lookupInBigHashTable_v3<<<gridDim,blockDim>>>(
			p_MBHashWrap->hashtable,
			p_MBHashWrap->next_pos,
			total_hits,
			p_scanMultiDBAuxWrap->offsetPairs,
			p_scanMultiDBAuxWrap->over_offset_pairs,
			p_scanMultiDBAuxWrap->over_hits_num,
			p_MBHashWrap->next_pos_len
			);
		getLastCudaError("kernel_lookupInBigHashTable_v3() execution failed.\n");
		slogfile.KernelEnd();
		slogfile.addTotalTime("lookup_kernel_time", slogfile.KernelElaplsedTime(), false );

		checkCudaErrors(cudaMemcpy(&total_hits, p_scanMultiDBAuxWrap->over_hits_num, sizeof(unsigned int), cudaMemcpyDeviceToHost));
		slogfile.addTotalNum("Kernel_lookupInBigHashTable hits", total_hits, false);
	}
	
	return total_hits;
}
Int4 
	s_gpu_MBScanSubject_11_2Mod4_scankernel_Opt_v3(const LookupTableWrap* lookup_wrap,
	const BLAST_SequenceBlk* subject,
	BlastOffsetPair* NCBI_RESTRICT offset_pairs, Int4 max_hits,  
	Int4* scan_range)
{
	BlastMBLookupTable* mb_lt = (BlastMBLookupTable*) lookup_wrap->lut;
	Uint4 total_hits = 0;

	max_hits -= mb_lt->longest_chain;
	ASSERT(lookup_wrap->lut_type == eMBLookupTable);
	ASSERT(mb_lt->lut_word_length == 11);
	if(scan_range[0] > scan_range[1]) return 0;

	
	Uint4 scan_range_temp = (scan_range[1]+mb_lt->lut_word_length - scan_range[0]);

	Uint4 subject_len = subject->length;
	int current_subject_id = subject->oid;

	GpuData* gpu_obj = BlastMGPUUtil.GetCurrentThreadGPUData();
	cudaScanAuxWrapMultiQueries* p_scanMultiDBAuxWrap = (cudaScanAuxWrapMultiQueries*) gpu_obj->m_global;
	cudaMBHashAuxWrap* p_MBHashWrap = (cudaMBHashAuxWrap*) gpu_obj->m_local;

	slogfile.Start();
	if (p_scanMultiDBAuxWrap->subject[current_subject_id] == NULL)
	{
		//printf("id:%d\n", p_scanMultiDBAuxWrap->subject_id);
		p_scanMultiDBAuxWrap->subject_id = current_subject_id;
		checkCudaErrors(cudaMalloc((void **)&p_scanMultiDBAuxWrap->subject[current_subject_id],(subject_len/4)));
		checkCudaErrors(cudaMemcpy(p_scanMultiDBAuxWrap->subject[current_subject_id], subject->sequence, (subject_len)/4 , cudaMemcpyHostToDevice));
	}
	slogfile.End();
	slogfile.addTotalTime("Scan CPU -> GPU Memory Time",slogfile.elaplsedTime(),false);

	total_hits = s_gpu_MBScanSubject_11_2Mod4_scankernel_Opt_v3_1(scan_range_temp, p_scanMultiDBAuxWrap, p_MBHashWrap, lookup_wrap,subject,offset_pairs,max_hits, scan_range);
	total_hits = s_gpu_MBScanSubject_11_2Mod4_scankernel_Opt_v3_2(total_hits, p_scanMultiDBAuxWrap, p_MBHashWrap,lookup_wrap,subject,offset_pairs,max_hits, scan_range);

	scan_range[0] = scan_range[1]+mb_lt->lut_word_length;

	return total_hits;
}


//////////////////////////////////////////////////////////////////////////
Int4 
	s_gpu_MBScanSubject_11_1Mod4_scankernel_Opt_v3_1(Uint4 scan_range_temp,
	cudaScanAuxWrapMultiQueries* p_scanMultiDBAuxWrap,
	cudaMBHashAuxWrap* p_MBHashWrap,
	const LookupTableWrap* lookup_wrap,
	const BLAST_SequenceBlk* subject,
	BlastOffsetPair* NCBI_RESTRICT offset_pairs, Int4 max_hits,  
	Int4* scan_range)
{

	BlastMBLookupTable* mb_lt = (BlastMBLookupTable*) lookup_wrap->lut;
	Uint4 total_hits = 0;
	Int4 top_shift =2; 
	Int4 pv_array_bts = mb_lt->pv_array_bts;

	int current_subject_id = subject->oid;

	checkCudaErrors(cudaMemset(p_scanMultiDBAuxWrap->total_hits, 0, sizeof(unsigned int)));  //初始化为0

	static int blocksize_x = SHARE_MEM_SIZE/2;
	dim3 blockSize(blocksize_x);
	dim3 gridSize;
	//scan_range_temp = (scan_range_temp/16) + (scan_range_temp%16);
	scan_range_temp = scan_range_temp/4;
	gridSize.x = (scan_range_temp+ blockSize.x -1)/ blockSize.x;

	static int max_grid_size = 16384;
	if (gridSize.x > max_grid_size)
	{
		gridSize.x = max_grid_size;
	}

	Uint4 global_size = gridSize.x;
	global_size *= blocksize_x;

	slogfile.KernelStart();
	//printf("%d %d\n", gridSize.x, blockSize.x);



	gpu_blastn_scan_11_1mod4_v3<<< gridSize, blockSize >>>(
	//gpu_blastn_scan_11_1mod4_opt<<< gridSize, blockSize >>>(
		(Uint1*)p_scanMultiDBAuxWrap->subject[current_subject_id], 
		p_scanMultiDBAuxWrap->offsetPairs, 
		p_scanMultiDBAuxWrap->total_hits, 
		scan_range_temp, 
		scan_range[0], 
		top_shift, 
		pv_array_bts,
		global_size,
		p_MBHashWrap->lookupArray); 

	getLastCudaError("gpu_blastn_scan_11_1mod4() execution failed.\n");

	slogfile.KernelEnd();
	slogfile.addTotalTime("scan_kernel_time", slogfile.KernelElaplsedTime(),false);

	checkCudaErrors(cudaMemcpy(&total_hits, p_scanMultiDBAuxWrap->total_hits, sizeof(Uint4), cudaMemcpyDeviceToHost));

	return total_hits;
}

//////////////////////////////////////////////////////////////////////////
Int4 
	s_gpu_MBScanSubject_11_1Mod4_scankernel_Opt_1(Uint4 scan_range_temp,
	cudaScanAuxWrapMultiQueries* p_scanMultiDBAuxWrap,
	cudaMBHashAuxWrap* p_MBHashWrap,
	const LookupTableWrap* lookup_wrap,
	const BLAST_SequenceBlk* subject,
	BlastOffsetPair* NCBI_RESTRICT offset_pairs, Int4 max_hits,  
	Int4* scan_range)
{

	BlastMBLookupTable* mb_lt = (BlastMBLookupTable*) lookup_wrap->lut;
	Uint4 total_hits = 0;
	Int4 top_shift =2; 
	Int4 pv_array_bts = mb_lt->pv_array_bts;

	int current_subject_id = subject->oid;

	checkCudaErrors(cudaMemset(p_scanMultiDBAuxWrap->total_hits, 0, sizeof(unsigned int)));  //初始化为0

	static int blocksize_x = BLOCK_SIZE_V1;
	dim3 blockSize(blocksize_x);
	dim3 gridSize;
	//scan_range_temp = (scan_range_temp/16) + (scan_range_temp%16);
	scan_range_temp = scan_range_temp/16;
	gridSize.x = (scan_range_temp-1/*+ blockSize.x -1*/)/blockSize.x;

	static int max_grid_size = 16384;
	if (gridSize.x > max_grid_size)
	{
		gridSize.x = max_grid_size;
	}

	Uint4 global_size = gridSize.x;
	global_size *= blocksize_x;

	slogfile.KernelStart();
	printf("%d %d\n", gridSize.x, blockSize.x);  
	gpu_blastn_scan_11_1mod4_opt_v1<<< gridSize, blockSize >>>(
		p_scanMultiDBAuxWrap->subject[current_subject_id], 
		p_scanMultiDBAuxWrap->offsetPairs, 
		p_scanMultiDBAuxWrap->total_hits, 
		scan_range_temp, 
		scan_range[0], 
		top_shift, 
		pv_array_bts,
		global_size,
		p_MBHashWrap->lookupArray); 

	getLastCudaError("gpu_blastn_scan_11_1mod4() execution failed.\n");

	slogfile.KernelEnd();
	slogfile.addTotalTime("scan_kernel_time", slogfile.KernelElaplsedTime(),false);

	checkCudaErrors(cudaMemcpy(&total_hits, p_scanMultiDBAuxWrap->total_hits, sizeof(Uint4), cudaMemcpyDeviceToHost));

	return total_hits;
}

Int4 
	s_gpu_MBScanSubject_11_1Mod4_scankernel_Opt_v3_2(int total_hits,
	cudaScanAuxWrapMultiQueries* p_scanMultiDBAuxWrap,
	cudaMBHashAuxWrap* p_MBHashWrap,
	const LookupTableWrap* lookup_wrap,
	const BLAST_SequenceBlk* subject,
	BlastOffsetPair* NCBI_RESTRICT offset_pairs, Int4 max_hits,  
	Int4* scan_range)
{
	if (total_hits >0)
	{
		Int4 threadNum = 512;
		Int4 blockNum = (total_hits + threadNum - 1)/threadNum;
		dim3 gridDim(blockNum, 1);
		dim3 blockDim(threadNum, 1);

		checkCudaErrors(cudaMemset(p_scanMultiDBAuxWrap->over_hits_num, 0, sizeof(unsigned int)));  //初始化为0

		slogfile.KernelStart();
		//cout << total_hits << endl; 

		kernel_lookupInBigHashTable_v3<<<gridDim,blockDim>>>(
			p_MBHashWrap->hashtable,
			p_MBHashWrap->next_pos,
			total_hits,
			p_scanMultiDBAuxWrap->offsetPairs,
			p_scanMultiDBAuxWrap->over_offset_pairs,
			p_scanMultiDBAuxWrap->over_hits_num,
			p_MBHashWrap->next_pos_len
			);
		getLastCudaError("kernel_lookupInBigHashTable_v3() execution failed.\n");
		slogfile.KernelEnd();
		slogfile.addTotalTime("lookup_kernel_time", slogfile.KernelElaplsedTime(), false );

		checkCudaErrors(cudaMemcpy(&total_hits, p_scanMultiDBAuxWrap->over_hits_num, sizeof(unsigned int), cudaMemcpyDeviceToHost));
		slogfile.addTotalNum("Kernel_lookupInBigHashTable hits", total_hits, false);


		slogfile.Start();
		checkCudaErrors(cudaMemcpy(offset_pairs, p_scanMultiDBAuxWrap->over_offset_pairs, total_hits* sizeof(BlastOffsetPair), cudaMemcpyDeviceToHost));	 	
		slogfile.End();
		slogfile.addTotalTime("GPU->CPU memory Time", slogfile.elaplsedTime(), false);
		thrust::sort(offset_pairs,offset_pairs+total_hits,OffsetPairCmp());
	}
	return total_hits;
}

Int4 
	s_gpu_MBScanSubject_11_1Mod4_scankernel_Opt_v3(const LookupTableWrap* lookup_wrap,
	const BLAST_SequenceBlk* subject,
	BlastOffsetPair* NCBI_RESTRICT offset_pairs, Int4 max_hits,  
	Int4* scan_range)
{
	BlastMBLookupTable* mb_lt = (BlastMBLookupTable*) lookup_wrap->lut;
	Uint4 total_hits = 0;

	max_hits -= mb_lt->longest_chain;
	ASSERT(lookup_wrap->lut_type == eMBLookupTable);
	ASSERT(mb_lt->lut_word_length == 11);
	if(scan_range[0] > scan_range[1]) return 0;


	Uint4 scan_range_temp = (scan_range[1]+mb_lt->lut_word_length - scan_range[0]);

	Uint4 subject_len = subject->length;
	int current_subject_id = subject->oid;

	GpuData* gpu_obj = BlastMGPUUtil.GetCurrentThreadGPUData();
	cudaScanAuxWrapMultiQueries* p_scanMultiDBAuxWrap = (cudaScanAuxWrapMultiQueries*) gpu_obj->m_global;
	cudaMBHashAuxWrap* p_MBHashWrap = (cudaMBHashAuxWrap*) gpu_obj->m_local;

	slogfile.Start();
	if (p_scanMultiDBAuxWrap->subject[current_subject_id] == NULL)
	{
		//cout <<"id:" << current_subject_id <<"\t";
		p_scanMultiDBAuxWrap->subject_id = current_subject_id;
		checkCudaErrors(cudaMalloc((void **)&p_scanMultiDBAuxWrap->subject[current_subject_id],(subject_len)/4));
		checkCudaErrors(cudaMemcpy(p_scanMultiDBAuxWrap->subject[current_subject_id], subject->sequence, (subject_len)/4 , cudaMemcpyHostToDevice));
	}
	slogfile.End();
	slogfile.addTotalTime("Scan CPU -> GPU Memory Time",slogfile.elaplsedTime(),false);

	total_hits = s_gpu_MBScanSubject_11_1Mod4_scankernel_Opt_v3_1(scan_range_temp, p_scanMultiDBAuxWrap, p_MBHashWrap, lookup_wrap,subject,offset_pairs,max_hits, scan_range);
	//total_hits = s_gpu_MBScanSubject_11_1Mod4_scankernel_Opt_1(scan_range_temp, p_scanMultiDBAuxWrap, p_MBHashWrap, lookup_wrap,subject,offset_pairs,max_hits, scan_range);
	total_hits = s_gpu_MBScanSubject_11_1Mod4_scankernel_Opt_v3_2(total_hits, p_scanMultiDBAuxWrap, p_MBHashWrap,lookup_wrap,subject,offset_pairs,max_hits, scan_range);

	scan_range[0] = scan_range[1]+mb_lt->lut_word_length;

	//exit(0); //
	return total_hits;
}
//////////////////////////////////////////////////////////////////////////


Int4 
	s_gpu_MBScanSubject_Any_scankernel_Opt_v3_1(Uint4 scan_range_temp,
	cudaScanAuxWrapMultiQueries* p_scanMultiDBAuxWrap,
	cudaMBHashAuxWrap* p_MBHashWrap,
	const LookupTableWrap* lookup_wrap,
	const BLAST_SequenceBlk* subject,
	BlastOffsetPair* NCBI_RESTRICT offset_pairs, Int4 max_hits,  
	Int4* scan_range)
{
	BlastMBLookupTable* mb_lt = (BlastMBLookupTable*) lookup_wrap->lut;
	Uint4 total_hits = 0;

	Int4 mask = mb_lt->hashsize - 1;
	max_hits -= mb_lt->longest_chain;

	Int4 shift = 2*(16 - (scan_range[0] % COMPRESSION_RATIO + mb_lt->lut_word_length));
	Uint4 pv_array_bts = mb_lt->pv_array_bts;

	int current_subject_id = subject->oid;

	static int blocksize_x = SHARE_MEM_SIZE/2;
	dim3 blockSize(blocksize_x);
	dim3 gridSize;
	scan_range_temp = scan_range_temp/16;
	gridSize.x = (scan_range_temp+ blockSize.x -1)/ blockSize.x;

	static int max_grid_size = 16384;
	if (gridSize.x > max_grid_size)
	{
		gridSize.x = max_grid_size;
	}
	Uint4 global_size = gridSize.x;
	global_size *= blocksize_x;

	slogfile.KernelStart();
	gpu_blastn_scan_Any_v3<<< gridSize, blockSize >>>(
		p_scanMultiDBAuxWrap->subject[current_subject_id], 
		p_scanMultiDBAuxWrap->offsetPairs, 
		p_scanMultiDBAuxWrap->total_hits, 
		scan_range_temp, 
		scan_range[0],
		mask,
		shift,
		pv_array_bts,
		global_size,
		p_MBHashWrap->lookupArray);
	getLastCudaError("gpu_blastn_scan_Any_v3() execution failed.\n");
	slogfile.KernelEnd();
	slogfile.addTotalTime("scan_kernel_time", slogfile.KernelElaplsedTime(),false);

	checkCudaErrors(cudaMemcpy(&total_hits, p_scanMultiDBAuxWrap->total_hits, sizeof(Uint4), cudaMemcpyDeviceToHost));
	
	return total_hits;
}
Int4 
	s_gpu_MBScanSubject_Any_scankernel_Opt_v3_2(int total_hits,
	cudaScanAuxWrapMultiQueries* p_scanMultiDBAuxWrap,
	cudaMBHashAuxWrap* p_MBHashWrap,
	const LookupTableWrap* lookup_wrap,
	const BLAST_SequenceBlk* subject,
	BlastOffsetPair* NCBI_RESTRICT offset_pairs, Int4 max_hits,  
	Int4* scan_range)
{
	if (total_hits > 0)
	{
		Int4 threadNum = 512;
		Int4 blockNum = (total_hits + threadNum - 1)/threadNum;
		dim3 gridDim(blockNum, 1);
		dim3 blockDim(threadNum, 1); 

		checkCudaErrors(cudaMemset(p_scanMultiDBAuxWrap->over_hits_num, 0, sizeof(unsigned int)));  //初始化为0

		slogfile.KernelStart();	 
		kernel_lookupInBigHashTable_v3<<<gridDim,blockDim>>>(
			p_MBHashWrap->hashtable,
			p_MBHashWrap->next_pos,
			total_hits,
			p_scanMultiDBAuxWrap->offsetPairs,
			p_scanMultiDBAuxWrap->over_offset_pairs,
			p_scanMultiDBAuxWrap->over_hits_num,
			p_MBHashWrap->next_pos_len
			);

		getLastCudaError("kernel_lookupInBigHashTable_v3() execution failed.\n");
		slogfile.KernelEnd();
		slogfile.addTotalTime("lookup_kernel_time", slogfile.KernelElaplsedTime(), false );
		checkCudaErrors(cudaMemcpy(&total_hits, p_scanMultiDBAuxWrap->over_hits_num, sizeof(unsigned int), cudaMemcpyDeviceToHost));
		slogfile.addTotalNum("Kernel_lookupInBigHashTable hits", total_hits, false);
	}
	return total_hits;
}
Int4 
	s_gpu_MBScanSubject_Any_scankernel_Opt_v3(const LookupTableWrap* lookup_wrap,
	const BLAST_SequenceBlk* subject,
	BlastOffsetPair* NCBI_RESTRICT offset_pairs, Int4 max_hits,  
	Int4* scan_range)
{
	BlastMBLookupTable* mb_lt = (BlastMBLookupTable*) lookup_wrap->lut;
	Uint4 total_hits = 0;
	max_hits -= mb_lt->longest_chain;
	ASSERT(lookup_wrap->lut_type == eMBLookupTable);
	ASSERT(mb_lt->lut_word_length == 11||
		mb_lt->lut_word_length == 12);
	if(scan_range[0] > scan_range[1]) return 0;

	Uint4 scan_range_temp = (scan_range[1] - scan_range[0]);

	Uint4 subject_len = subject->length; //length is bp, 2bit.
	int current_subject_id = subject->oid;

	GpuData* gpu_obj = BlastMGPUUtil.GetCurrentThreadGPUData();
	cudaScanAuxWrapMultiQueries* p_scanMultiDBAuxWrap = (cudaScanAuxWrapMultiQueries*) gpu_obj->m_global;
	cudaMBHashAuxWrap* p_MBHashWrap = (cudaMBHashAuxWrap*) gpu_obj->m_local;

	slogfile.Start();
	if (p_scanMultiDBAuxWrap->subject[current_subject_id] == NULL)
	{	
		p_scanMultiDBAuxWrap->subject_id = current_subject_id;
		//printf("id:%d\n", p_scanMultiDBAuxWrap->subject_id);
		checkCudaErrors(cudaMalloc((void **)&p_scanMultiDBAuxWrap->subject[current_subject_id],(subject_len/4)));
		checkCudaErrors(cudaMemcpy(p_scanMultiDBAuxWrap->subject[current_subject_id], subject->sequence, (subject_len)/4 , cudaMemcpyHostToDevice));
	}
	checkCudaErrors(cudaMemset(p_scanMultiDBAuxWrap->total_hits, 0, sizeof(unsigned int)));  //初始化为0

	slogfile.End();
	slogfile.addTotalTime("Scan CPU -> GPU Memory Time",slogfile.elaplsedTime(),false);

	total_hits = s_gpu_MBScanSubject_Any_scankernel_Opt_v3_1(scan_range_temp, p_scanMultiDBAuxWrap, p_MBHashWrap,lookup_wrap,subject,offset_pairs,max_hits,scan_range);
	total_hits = s_gpu_MBScanSubject_Any_scankernel_Opt_v3_2(total_hits, p_scanMultiDBAuxWrap, p_MBHashWrap,lookup_wrap,subject,offset_pairs,max_hits,scan_range);

	scan_range[0] = scan_range[1]+mb_lt->lut_word_length;

	return total_hits;
}

//////////////////////////////////////////////////////////////////////////
///MiniExtend
//////////////////////////////////////////////////////////////////////////

Int4 
	s_gpu_BlastNaExtend_Opt_v3(BlastOffsetPair * offset_pairs, Int4 num_hits,
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
	int current_subject_id = subject->oid;

	unsigned int exact_hits_num = 0;

	Int4 threadNum = 256;
	Int4 blockNum_Ex = (num_hits + threadNum - 1)/threadNum;
	dim3 gridDim_Ex(blockNum_Ex, 1);
	dim3 blockDim_Ex(threadNum, 1);

	//printf("%d, %d\n", threadNum, blockNum_Ex);
	GpuData* gpu_obj = BlastMGPUUtil.GetCurrentThreadGPUData();
	cudaScanAuxWrapMultiQueries* p_scanMultiDBAuxWrap = (cudaScanAuxWrapMultiQueries*) gpu_obj->m_global;
	cudaMBHashAuxWrap* p_MBHashWrap = (cudaMBHashAuxWrap*) gpu_obj->m_local;

	checkCudaErrors(cudaMemset(p_scanMultiDBAuxWrap->over_hits_num, 0, sizeof(unsigned int)));  //初始化为0

	Int4 ext_to =  word_length - lut_word_length;

	//printf("id:%d,num_his:%d \n", p_scanMultiDBAuxWrap->subject_id,num_hits);
	//printf("query length:%d\n", query->length);

	slogfile.KernelStart();

	kernel_s_BlastNaExtend_withoutHash_v3<<<gridDim_Ex, blockDim_Ex>>> (
		(Uint1*)p_scanMultiDBAuxWrap->subject[current_subject_id],
		p_MBHashWrap->query,
		lut_word_length,
		ext_to,		
		num_hits,
		p_scanMultiDBAuxWrap->over_offset_pairs, p_scanMultiDBAuxWrap->offsetPairs, 
		p_scanMultiDBAuxWrap->over_hits_num, 
		s_range,
		query->length);

	getLastCudaError("kernel_s_BlastNaExtend_withoutHash() v1 execution failed.\n");

	slogfile.KernelEnd();
	slogfile.addTotalTime("extend_kernel_time", slogfile.KernelElaplsedTime(), false);

	checkCudaErrors(cudaMemcpy(&exact_hits_num, p_scanMultiDBAuxWrap->over_hits_num, sizeof(unsigned int), cudaMemcpyDeviceToHost));
	slogfile.addTotalNum("mini_extended hits", exact_hits_num,false);

	if (exact_hits_num >0)
	{
		//cout << "Thread: " << GetCurrentThreadId() << "offset_address" << offset_pairs << endl;
		slogfile.Start();
		checkCudaErrors(cudaMemcpy(offset_pairs, p_scanMultiDBAuxWrap->offsetPairs, exact_hits_num* sizeof(BlastOffsetPair), cudaMemcpyDeviceToHost));	 	
		slogfile.End();
		slogfile.addTotalTime("GPU->CPU memory Time", slogfile.elaplsedTime(), false);
		thrust::sort(offset_pairs,offset_pairs+exact_hits_num,OffsetPairCmp());
	}

	__int64 c1 = slogfile.Start();
	if (exact_hits_num >0)
	{  
		if (word_params->container_type == eDiagHash) {
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
		else{
			for(int i=0; i<exact_hits_num; i++){
				Int4 s_offset = offset_pairs[i].qs_offsets.s_off;
				Int4 q_offset = offset_pairs[i].qs_offsets.q_off;	
				hits_extended += s_BlastnDiagTableExtendInitialHit(query, subject, 
					q_offset, s_offset,  
					lut->masked_locations, 
					query_info, s_range, 
					word_length, lut_word_length,
					lookup_wrap,
					word_params, matrix,
					ewp->diag_table,
					init_hitlist);
			}
		}
	}


	__int64 c2 = slogfile.End();
	slogfile.addTotalTime("Hits extend time",c1,c2, false);

	return hits_extended; 
}

//////////////////////////////////////////////////////////////////////////
//For blastn extend
Int4
	s_new_BlastNaExtendDirect(BlastOffsetPair * offset_pairs, Int4 num_hits,
	const BlastInitialWordParameters * word_params,
	LookupTableWrap * lookup_wrap,
	BLAST_SequenceBlk * query,
	BLAST_SequenceBlk * subject, Int4 ** matrix,
	BlastQueryInfo * query_info,
	Blast_ExtendWord * ewp,
	BlastInitHitList * init_hitlist,
	Uint4 s_range)
{
	Int4 index = 0;
	Int4 hits_extended = 0;
	Int4 word_length;

	if (lookup_wrap->lut_type == eMBLookupTable) {
		BlastMBLookupTable *lut = (BlastMBLookupTable *) lookup_wrap->lut;
		word_length = (lut->discontiguous) ? lut->template_length : lut->word_length;
		ASSERT(word_length == lut->lut_word_length || lut->discontiguous);
	} 
	else if (lookup_wrap->lut_type == eSmallNaLookupTable) {
		BlastSmallNaLookupTable *lut = 
			(BlastSmallNaLookupTable *) lookup_wrap->lut;
		word_length = lut->word_length;
	} 
	else {
		BlastNaLookupTable *lut = (BlastNaLookupTable *) lookup_wrap->lut;
		word_length = lut->word_length;
	}

	if (word_params->container_type == eDiagHash) {
		for (; index < num_hits; ++index) {
			Int4 s_offset = offset_pairs[index].qs_offsets.s_off;
			Int4 q_offset = offset_pairs[index].qs_offsets.q_off;

			hits_extended += s_BlastnDiagHashExtendInitialHit(query, subject, 
				q_offset, s_offset,  
				NULL,
				query_info, s_range, 
				word_length, word_length,
				lookup_wrap,
				word_params, matrix,
				ewp->hash_table,
				init_hitlist);
		}
	} 
	else {
		for (; index < num_hits; ++index) {
			Int4 s_offset = offset_pairs[index].qs_offsets.s_off;
			Int4 q_offset = offset_pairs[index].qs_offsets.q_off;

			hits_extended += s_BlastnDiagTableExtendInitialHit(query, subject, 
				q_offset, s_offset,  
				NULL,
				query_info, s_range, 
				word_length, word_length,
				lookup_wrap,
				word_params, matrix,
				ewp->diag_table,
				init_hitlist);
		}
	}
	return hits_extended;
}
