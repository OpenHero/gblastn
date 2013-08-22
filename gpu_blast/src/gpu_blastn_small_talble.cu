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

#include <algo/blast/gpu_blast/gpu_blastn_small_table_setting.h>


#include "gpu_blastn_ungapped_extension_functions.h"
#include "gpu_blastn_small_table_kernel.cuh"

#include <algo/blast/gpu_blast/gpu_logfile.h>

#if USE_OPENMP
#include <omp.h>
#endif


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
typedef struct cudaSmallTableAuxWrap {
	Int2 * overflowtable;
	Uint4 overflowtable_len;
	Uint1 * query_compressed_nuc_seq_start;
	BlastContextInfo* contextinfo;
	Int2 * backbone;
} cudaSmallTableAuxWrap;

//////////////////////////////////////////////////////////////////////////
//multi queries
//////////////////////////////////////////////////////////////////////////
static cudaScanAuxWrapMultiQueries cuda_scanMultiAuxWrap;
static cudaSmallTableAuxWrap cuda_smallAuxWrap;

bool isInitScanSmallGPUMem = false;

//////////////////////////////////////////////////////////////////////////
//init database gpu memory
void InitScanSmallGPUMem(const BlastSeqSrc* seq_src)
{
	Uint4* d_total_hits = NULL;
	Uint4* d_over_total_hits = NULL;
	BlastOffsetPair* d_offsetPairs = NULL;
	BlastOffsetPair* d_over_offsetPairs = NULL;
	int subject_num = BlastSeqSrcGetNumSeqs(seq_src);
	Uint4** d_subject = new Uint4*[subject_num]();
	
	checkCudaErrors(cudaMalloc((void **)&d_total_hits,sizeof(Uint4)));
	checkCudaErrors(cudaMalloc((void **)&d_over_total_hits, sizeof(Uint4)));
	checkCudaErrors(cudaMalloc((void **)&d_offsetPairs, OFFSET_ARRAY_SIZE * sizeof(BlastOffsetPair)));
	checkCudaErrors(cudaMalloc((void **)&d_over_offsetPairs, OFFSET_ARRAY_SIZE * sizeof(BlastOffsetPair)));

	cuda_scanMultiAuxWrap.offsetPairs = d_offsetPairs;
	cuda_scanMultiAuxWrap.over_offset_pairs = d_over_offsetPairs;
	cuda_scanMultiAuxWrap.total_hits = d_total_hits;
	cuda_scanMultiAuxWrap.over_hits_num = d_over_total_hits;

	cuda_scanMultiAuxWrap.subject_id = 0;
	cuda_scanMultiAuxWrap.subject = d_subject;
	for (int i = 0; i < subject_num; i ++)
	{
		cuda_scanMultiAuxWrap.subject[i]= NULL;
	}
}

extern "C" void InitSmallGPUMem(const BlastSeqSrc* seq_src)
{
	if (isInitScanSmallGPUMem == false)
	{
		InitScanSmallGPUMem(seq_src);
	}
	isInitScanSmallGPUMem = true;
}

extern "C" void ReleaseScanSmallGPUMem()
{
	if (isInitScanSmallGPUMem == true)
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
	isInitScanSmallGPUMem = false;
}

extern "C" void SetSmallTableSubjectID(int id)
{
	cuda_scanMultiAuxWrap.subject_id = id;
}

//////////////////////////////////////////////////////////////////////////
//init query gpu memory


void InitSmallNaLookupTableTexture(LookupTableWrap* lookup_wrap)
{
	BlastSmallNaLookupTable *lookup = (BlastSmallNaLookupTable *) lookup_wrap->lut;
	Int4 backbone_size = lookup->backbone_size;

	Int2* d_backbone = NULL;
	//create cuda
	checkCudaErrors(cudaMalloc( (void**)&d_backbone, backbone_size * sizeof (Int2)));
	checkCudaErrors(cudaMemcpy( d_backbone, lookup->final_backbone, backbone_size * sizeof (Int2), cudaMemcpyHostToDevice));

	SET_INT2_BASE;

	cuda_smallAuxWrap.backbone = d_backbone;
}

//extern "C" void CopySmallNaLookupTableTexture(LookupTableWrap* lookup_wrap, BLAST_SequenceBlk* query, BlastQueryInfo* query_info)
extern "C" void InitQueryGPUMem(LookupTableWrap* lookup_wrap, BLAST_SequenceBlk* query, BlastQueryInfo* query_info)
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
	

	cuda_smallAuxWrap.contextinfo = d_contextinfo;
	cuda_smallAuxWrap.query_compressed_nuc_seq_start = d_query_compressed_nuc_seq_start;
	cuda_smallAuxWrap.overflowtable = d_overflow;
	cuda_smallAuxWrap.overflowtable_len = overflow_size;

	InitSmallNaLookupTableTexture(lookup_wrap);
}

void FreeSmallNaLookupTableTexture()
{
	checkCudaErrors(cudaUnbindTexture(tx_backbone));
	checkCudaErrors(cudaFree(cuda_smallAuxWrap.backbone));
}
//////////////////////////////////////////////////////////////////////////

void ReleaseQueryGPUMem()
{
	checkCudaErrors(cudaFree(cuda_smallAuxWrap.contextinfo));
	checkCudaErrors(cudaFree(cuda_smallAuxWrap.query_compressed_nuc_seq_start));
	checkCudaErrors(cudaFree(cuda_smallAuxWrap.overflowtable));
	FreeSmallNaLookupTableTexture();
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
	s_gpu_MBScanSubject_8_1Mod4_scankernel_Opt(const LookupTableWrap* lookup_wrap,
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

	Uint4 scan_range_temp = (scan_range[1]+11 - scan_range[0]);

	slogfile.Start();
	if (cuda_scanMultiAuxWrap.subject[cuda_scanMultiAuxWrap.subject_id] == NULL)
	{
		//printf("id:%d\n", cuda_scanMultiAuxWrap.subject_id);
		checkCudaErrors(cudaMalloc((void **)&cuda_scanMultiAuxWrap.subject[cuda_scanMultiAuxWrap.subject_id],((scan_range_temp)+3)/4));
		checkCudaErrors(cudaMemcpy(cuda_scanMultiAuxWrap.subject[cuda_scanMultiAuxWrap.subject_id], subject->sequence, (scan_range_temp) / 4 , cudaMemcpyHostToDevice));
	}
	slogfile.End();
	slogfile.addTotalTime("Scan CPU -> GPU Memory Time",slogfile.elaplsedTime(), false);
	 	
	checkCudaErrors(cudaMemset(cuda_scanMultiAuxWrap.total_hits, 0, sizeof(unsigned int)));  //初始化为0
	checkCudaErrors(cudaMemset(cuda_scanMultiAuxWrap.over_hits_num, 0, sizeof(unsigned int)));  //初始化为0
	
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

	gpu_blastn_scan_8_1mod4<<< gridSize, blockSize >>>(
		cuda_scanMultiAuxWrap.subject[cuda_scanMultiAuxWrap.subject_id],
		cuda_scanMultiAuxWrap.offsetPairs,
		cuda_scanMultiAuxWrap.over_offset_pairs,
		cuda_scanMultiAuxWrap.total_hits,
		cuda_scanMultiAuxWrap.over_hits_num,
		scan_range_temp, 
		scan_range[0],
		global_size,
		cuda_smallAuxWrap.backbone);


	getLastCudaError("gpu_blastn_scan_8_1mod4() execution failed.\n");

	slogfile.KernelEnd();
	slogfile.addTotalTime("Scan Kernel Time", slogfile.KernelElaplsedTime(), false);

	checkCudaErrors(cudaMemcpy(&total_hits, cuda_scanMultiAuxWrap.over_hits_num, sizeof(Uint4), cudaMemcpyDeviceToHost));

	//printf("total_hits:%d\n",total_hits);
	//total_hits = 0;

	if (total_hits > 0)
	{
		Int4 threadNum = 128;
		Int4 blockNum = (total_hits + threadNum - 1)/threadNum;
		dim3 gridDim(blockNum, 1);
		dim3 blockDim(threadNum, 1);

		//checkCudaErrors(cudaMemset(cuda_extenWrap.exact_hits_num, 0, sizeof(unsigned int)));  //初始化为0

		slogfile.KernelStart();

		kernel_lookupInSmallTable<<<gridDim,blockDim>>>(
			cuda_smallAuxWrap.overflowtable,
			total_hits,
			cuda_scanMultiAuxWrap.over_offset_pairs,
			cuda_scanMultiAuxWrap.offsetPairs,
			cuda_scanMultiAuxWrap.total_hits,
			cuda_smallAuxWrap.overflowtable_len
			);
		getLastCudaError("kernel_lookupInsSmallTable() execution failed.\n");
		slogfile.KernelEnd();
		slogfile.addTotalTime("LookUpTableHash Time v1", slogfile.KernelElaplsedTime() ,false );
	}
	checkCudaErrors(cudaMemcpy(&total_hits, cuda_scanMultiAuxWrap.total_hits, sizeof(unsigned int), cudaMemcpyDeviceToHost));
	slogfile.addTotalNum("Kernel_lookupInBigHashTable hits", total_hits,false);
	//printf("bighash_hits:%d\n", total_hits);
	
	//checkCudaErrors(cudaMemcpy(offset_pairs, cuda_scanMultiAuxWrap.offsetPairs, total_hits * sizeof(BlastOffsetPair), cudaMemcpyDeviceToHost));
	
	//slogfile.m_file << "Scan hits:"<<total_hits<<endl;

	//for (int i = 0; i < total_hits; i++)
	//{
	//	printf("%d:%d ; ",offset_pairs[i].qs_offsets.q_off,offset_pairs[i].qs_offsets.s_off);
	//}
	//exit(0);

	scan_range[0] = scan_range[1]+11;

	return total_hits;
}

extern "C" Int4 
	s_gpu_BlastSmallExtend(BlastOffsetPair * offset_pairs, Int4 num_hits,
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
	
	unsigned int exact_hits_num = 0;

	Int4 threadNum = 256;
	Int4 blockNum_Ex = (num_hits + threadNum - 1)/threadNum;
	dim3 gridDim_Ex(blockNum_Ex, 1);
	dim3 blockDim_Ex(threadNum, 1);

	checkCudaErrors(cudaMemset(cuda_scanMultiAuxWrap.over_hits_num, 0, sizeof(unsigned int)));  //初始化为0

	//printf("id:%d,num_his:%d \n", cuda_scanMultiAuxWrap.subject_id,num_hits);
	slogfile.KernelStart();

	kernel_s_BlastSmallExtend<<<gridDim_Ex, blockDim_Ex>>> (
		(Uint1*)cuda_scanMultiAuxWrap.subject[cuda_scanMultiAuxWrap.subject_id],
		cuda_smallAuxWrap.query_compressed_nuc_seq_start,
		word_length,
		lut_word_length,	
		num_hits,
		cuda_scanMultiAuxWrap.offsetPairs,cuda_scanMultiAuxWrap.over_offset_pairs,  
		cuda_scanMultiAuxWrap.over_hits_num, 
		s_range,
		cuda_smallAuxWrap.contextinfo,
		query_info->last_context);
	getLastCudaError("kernel_s_BlastSmallExtend() execution failed.\n");

	slogfile.KernelEnd();
	slogfile.addTotalTime("kernel_s_BlastSmallExtend Time", slogfile.KernelElaplsedTime(), false);

	checkCudaErrors(cudaMemcpy(&exact_hits_num, cuda_scanMultiAuxWrap.over_hits_num, sizeof(unsigned int), cudaMemcpyDeviceToHost));
	slogfile.addTotalNum("Small_extended hits", exact_hits_num,false);

	//printf("hits:%d\n", exact_hits_num);


//#if 0
//
//		checkCudaErrors(cudaMemcpy(offset_pairs, cuda_smallAuxWrap.over_offset_pairs, 3*sizeof(BlastOffsetPair), cudaMemcpyDeviceToHost));
//
//		if ((offset_pairs[0].qs_offsets.q_off == 641))
//		{
//
//			printf("%d:%d:%d:%d:%d:%d \n ",
//				offset_pairs[0].qs_offsets.s_off,offset_pairs[0].qs_offsets.q_off,
//				offset_pairs[1].qs_offsets.s_off,offset_pairs[1].qs_offsets.q_off,
//				offset_pairs[2].qs_offsets.s_off,offset_pairs[2].qs_offsets.q_off);
//		}
//	
//#endif


	if (exact_hits_num >0)
	{
		slogfile.Start();
		checkCudaErrors(cudaMemcpy(offset_pairs, cuda_scanMultiAuxWrap.over_offset_pairs, exact_hits_num * sizeof(BlastOffsetPair), cudaMemcpyDeviceToHost));
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


	__int64 c2 = slogfile.End();
	slogfile.addTotalTime("Hits extend time",c1,c2, false);

	return hits_extended; 
}

//////////////////////////////////////////////////////////////////////////
//cpu


/** 
* Copy query offsets from a BlastSmallNaLookupTable
* @param offset_pairs A pointer into the destination array. [out]
* @param index The index value of the word to retrieve. [in]
* @param s_off The subject offset to be associated with the retrieved query offset(s). [in]
* @param total_hits The total number of query offsets save thus far [in]
* @param overflow Packed list of query offsets [in]
* @return Number of hits saved
*/
static NCBI_INLINE Int4 s_BlastSmallNaRetrieveHits(
	BlastOffsetPair * NCBI_RESTRICT offset_pairs,
	Int4 index, Int4 s_off,
	Int4 total_hits, Int2 *overflow)
{
	if (index >= 0) {
		offset_pairs[total_hits].qs_offsets.q_off = index;
		offset_pairs[total_hits].qs_offsets.s_off = s_off;
		return 1;
	}
	else {
		Int4 num_hits = 0;
		Int4 src_off = -index;
		index = overflow[src_off++];
		do {
			offset_pairs[total_hits+num_hits].qs_offsets.q_off = index;
			offset_pairs[total_hits+num_hits].qs_offsets.s_off = s_off;
			num_hits++;
			index = overflow[src_off++];
		} while (index >= 0);

		return num_hits;
	}
}

/** access the small-query lookup table */
#define SMALL_NA_ACCESS_HITS(x)                                 \
	if (index != -1) {                                          \
	if (total_hits > max_hits) {                            \
	scan_range[0] += (x);                                       \
	break;                                              \
	}                                                       \
	total_hits += s_BlastSmallNaRetrieveHits(offset_pairs,  \
	index, scan_range[0] + (x),  \
	total_hits,    \
	overflow);     \
	}


extern "C" Int4 s_new_BlastSmallNaScanSubject_8_1Mod4(
	const LookupTableWrap * lookup_wrap,
	const BLAST_SequenceBlk * subject,
	BlastOffsetPair * NCBI_RESTRICT offset_pairs,
	Int4 max_hits, Int4 * scan_range)
{
	BlastSmallNaLookupTable *lookup = 
		(BlastSmallNaLookupTable *) lookup_wrap->lut;
	const Int4 kLutWordLength = 8;
	const Int4 kLutWordMask = (1 << (2 * kLutWordLength)) - 1;
	//Int4 scan_step = lookup->scan_step;
	Int4 scan_step = 16;
	Int4 scan_step_byte = scan_step / COMPRESSION_RATIO;
	Uint1 *s = subject->sequence + scan_range[0] / COMPRESSION_RATIO;
	Int4 total_hits = 0;
	Int2 *backbone = lookup->final_backbone;
	Int2 *overflow = lookup->overflow;
	Int4 index; 

	max_hits -= lookup->longest_chain;
	ASSERT(lookup_wrap->lut_type == eSmallNaLookupTable);
	ASSERT(lookup->lut_word_length == 8);
	ASSERT(lookup->scan_step % COMPRESSION_RATIO == 1);

	//switch (scan_range[0] % COMPRESSION_RATIO) {
	//case 1: goto base_1;
	//case 2: goto base_2;
	//case 3: goto base_3;
	//}

	while (scan_range[0] <= scan_range[1]) {

		index = s[0] << 8 | s[1];
		s += scan_step_byte;
		index = backbone[index];
		SMALL_NA_ACCESS_HITS(0);
		scan_range[0] += scan_step;

//base_1:
//		if (scan_range[0] > scan_range[1])
//			break;
//
//		index = s[0] << 16 | s[1] << 8 | s[2];
//		s += scan_step_byte;
//		index = backbone[(index >> 6) & kLutWordMask];
//		SMALL_NA_ACCESS_HITS(0);
//		scan_range[0] += scan_step;
//
//base_2:
//		if (scan_range[0] > scan_range[1])
//			break;
//
//		index = s[0] << 16 | s[1] << 8 | s[2];
//		s += scan_step_byte;
//		index = backbone[(index >> 4) & kLutWordMask];
//		SMALL_NA_ACCESS_HITS(0);
//		scan_range[0] += scan_step;
//
//base_3:
//		if (scan_range[0] > scan_range[1])
//			break;
//
//		index = s[0] << 16 | s[1] << 8 | s[2];
//		s += scan_step_byte + 1;
//		index = backbone[(index >> 2) & kLutWordMask];
//		SMALL_NA_ACCESS_HITS(0);
//		scan_range[0] += scan_step;
	}
	return total_hits;
}

//////////////////////////////////////////////////////////////////////////
/** Entry i of this list gives the number of pairs of
 * bits that are zero in the bit pattern of i, looking 
 * from right to left
 */
Uint1 s_ExactMatchExtendLeft[256] = {
4, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 
2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 
2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 
2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 
3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 
2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 
2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 
2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 
3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 
2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 
2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 
2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 
3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 
2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 
2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 
2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 
};

/** Entry i of this list gives the number of pairs of
 * bits that are zero in the bit pattern of i, looking 
 * from left to right
 */
Uint1 s_ExactMatchExtendRight[256] = {
4, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
};
/** Perform exact match extensions on the hits retrieved from
 * small-query blastn lookup tables, assuming an arbitrary number of bases 
 * in a lookup and arbitrary start offset of each hit. Also 
 * update the diagonal structure.
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
 * @return Number of hits extended. 
 */
extern "C"  Int4
s_new_BlastSmallNaExtend(const BlastOffsetPair * offset_pairs, Int4 num_hits,
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
    BlastSmallNaLookupTable *lut = (BlastSmallNaLookupTable *) lookup_wrap->lut;
    Int4 word_length = lut->word_length; 
    Int4 lut_word_length = lut->lut_word_length; 
    Uint1 *q = query->compressed_nuc_seq;
    Uint1 *s = subject->sequence;

	Int4 t_s_offset;
	Int4 t_q_offset;
	Int4 t_context;
	Int4 t_q_start;
	Int4 t_q_range;
	Int4 t_ext_max;


    for (; index < num_hits; ++index) {
        Int4 s_offset = offset_pairs[index].qs_offsets.s_off;
        Int4 q_offset = offset_pairs[index].qs_offsets.q_off;
        Int4 s_off;
        Int4 q_off;
        Int4 ext_left = 0;
        Int4 ext_right = 0;
        Int4 context = BSearchContextInfo(q_offset, query_info);
        Int4 q_start = query_info->contexts[context].query_offset;
        Int4 q_range = q_start + query_info->contexts[context].query_length;
        Int4 ext_max = MIN(MIN(word_length - lut_word_length, s_offset), q_offset - q_start);

		 t_s_offset = s_offset;
		 t_q_offset = q_offset;
		 t_context = context;
		 t_q_start = q_start;
		 t_q_range = q_range;
		 t_ext_max = ext_max;


		//printf("%d:%d:%d:%d:%d:%d \n ",t_s_offset, t_q_offset, t_context, t_q_start, t_q_range, t_ext_max);		

        /* Start the extension at the first multiple of 4 bases in
           the subject sequence to the right of the seed.
           Collect exact matches in groups of four, until a
           mismatch is encountered or the expected number of
           matches is found. The index into q[] below can
           technically be negative, but the compressed version
           of the query has extra pad bytes before q[0] */

        Int4 rsdl = COMPRESSION_RATIO - (s_offset % COMPRESSION_RATIO);
        s_offset += rsdl;
        q_offset += rsdl;
        ext_max  += rsdl;

        s_off = s_offset;
        q_off = q_offset;

        while (ext_left < ext_max) {
            Uint1 q_byte = q[q_off - 4];
            Uint1 s_byte = s[s_off / COMPRESSION_RATIO - 1];
            Uint1 bases = s_ExactMatchExtendLeft[q_byte ^ s_byte];
            ext_left += bases;
            if (bases < 4)
                break;
            q_off -= 4;
            s_off -= 4;
        }
        ext_left = MIN(ext_left, ext_max);

        /* extend to the right. The extension begins at the first
           base not examined by the left extension */

        s_off = s_offset;
        q_off = q_offset;
        ext_max = MIN(MIN(word_length - ext_left, s_range - s_off), q_range - q_off);
        while (ext_right < ext_max) {
            Uint1 q_byte = q[q_off];
            Uint1 s_byte = s[s_off / COMPRESSION_RATIO];
            Uint1 bases = s_ExactMatchExtendRight[q_byte ^ s_byte];
            ext_right += bases;
            if (bases < 4)
                break;
            q_off += 4;
            s_off += 4;
        }
        ext_right = MIN(ext_right, ext_max);

        if (ext_left + ext_right < word_length)
            continue;

        q_offset -= ext_left;
        s_offset -= ext_left;
        
		printf("%d:%d:%d:%d:%d:%d \n ",t_s_offset, t_q_offset, t_context, t_q_start, t_q_range, t_ext_max);

        if (word_params->container_type == eDiagHash) {
            hits_extended += s_BlastnDiagHashExtendInitialHit(query, subject, 
                                                q_offset, s_offset,  
                                                lut->masked_locations, 
                                                query_info, s_range, 
                                                word_length, lut_word_length,
                                                lookup_wrap,
                                                word_params, matrix,
                                                ewp->hash_table,
                                                init_hitlist);
        } else {
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
    return hits_extended;
}