#ifndef __GPU_BLASTN_ALL_SUBJECT_IN_ONE_H__
#define __GPU_BLASTN_ALL_SUBJECT_IN_ONE_H__

#include <algo/blast/gpu_blast/gpu_blastn_config.hpp>
#include <algo/blast/gpu_blast/gpu_blastn.h>
//#include "gpu_blastn_OneVol_DB.h"

#ifdef __cplusplus
extern "C"{
#endif

void initAllGPUMemoryVolume(
	const BlastSeqSrc* seq_src, 
	LookupTableWrap* lookup_wrap,
	BLAST_SequenceBlk* query);

void freeAllGPUMemory();

int getVolumeNum(void* seq_src);

Int4 s_gpu_MBScanSubject_11_2Mod4(const LookupTableWrap* lookup_wrap,
	const BLAST_SequenceBlk* subject,
	BlastOffsetPair* NCBI_RESTRICT offset_pairs, Int4 max_hits,  
	Int4* scan_range);

Int4 s_gpu_MBScanSubject_11_2Mod4_OneVol_v1(const LookupTableWrap* lookup_wrap,
	const BLAST_SequenceBlk* subject,
	BlastOffsetPair* NCBI_RESTRICT offset_pairs, Int4 max_hits,  
	Int4* scan_range);

Int4 s_gpu_BlastNaExtend_OneVol_v1(const BlastOffsetPair * offset_pairs, Int4 num_hits,
	const BlastInitialWordParameters * word_params,
	LookupTableWrap * lookup_wrap,
	BLAST_SequenceBlk * query,
	BLAST_SequenceBlk * subject, Int4 ** matrix,
	BlastQueryInfo * query_info,
	Blast_ExtendWord * ewp,
	BlastInitHitList * init_hitlist,
	Uint4 s_range );
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C"{
#endif

	void initAllGPUMemory( 
		LookupTableWrap* lookup_wrap,
		BLAST_SequenceBlk* query);

	void freeAllGPUMemory();   //gpu_blastn added by kyzhao

	Int4 s_gpu_MBScanSubject_11_2Mod4(const LookupTableWrap* lookup_wrap,
		const BLAST_SequenceBlk* subject,
		BlastOffsetPair* NCBI_RESTRICT offset_pairs, Int4 max_hits,  
		Int4* scan_range);

	Int4 s_gpu_MBScanSubject_11_2Mod4_v1(const LookupTableWrap* lookup_wrap,
		const BLAST_SequenceBlk* subject,
		BlastOffsetPair* NCBI_RESTRICT offset_pairs, Int4 max_hits,  
		Int4* scan_range);

	Int4 
		s_gpu_MBScanSubject_11_2Mod4_v1_scankernel_v1(const LookupTableWrap* lookup_wrap,
		const BLAST_SequenceBlk* subject,
		BlastOffsetPair* NCBI_RESTRICT offset_pairs, Int4 max_hits,  
		Int4* scan_range);

	Int4 
		s_gpu_MBScanSubject_Any_scankernel_v1(const LookupTableWrap* lookup_wrap,
		const BLAST_SequenceBlk* subject,
		BlastOffsetPair* NCBI_RESTRICT offset_pairs, Int4 max_hits,  
		Int4* scan_range);

	//////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////

	Int4 s_gpu_BlastNaExtend_v1(const BlastOffsetPair * offset_pairs, Int4 num_hits,
		const BlastInitialWordParameters * word_params,
		LookupTableWrap * lookup_wrap,
		BLAST_SequenceBlk * query,
		BLAST_SequenceBlk * subject, Int4 ** matrix,
		BlastQueryInfo * query_info,
		Blast_ExtendWord * ewp,
		BlastInitHitList * init_hitlist,
		Uint4 s_range );

	Int4 
		s_gpu_BlastNaExtend_v2(const BlastOffsetPair * offset_pairs, Int4 num_hits,
		const BlastInitialWordParameters * word_params,
		LookupTableWrap * lookup_wrap,
		BLAST_SequenceBlk * query,
		BLAST_SequenceBlk * subject, Int4 ** matrix,
		BlastQueryInfo * query_info,
		Blast_ExtendWord * ewp,
		BlastInitHitList * init_hitlist,
		Uint4 s_range );

	/************************************************************************/
	/* CUDA implementation of ungapped extensnsion, added by qjli           */
	/************************************************************************/
	Int4 s_gpu_BlastNaExtend(const BlastOffsetPair * offset_pairs,
		Int4 num_hits,
		const BlastInitialWordParameters * word_params,
		LookupTableWrap * lookup_wrap,
		BLAST_SequenceBlk * query,
		BLAST_SequenceBlk * subject,
		Int4 ** matrix,
		BlastQueryInfo * query_info,
		Blast_ExtendWord * ewp,
		BlastInitHitList * init_hitlist,
		Uint4 s_range );

	//////////////////////////////////////////////////////////////////////////
	//multi queries
	//////////////////////////////////////////////////////////////////////////
	//extern cudaScanAuxWrapMultiQueries cuda_scanMultiAuxWrap;
	//void InitGPUMem(const BlastSeqSrc* seq_src, LookupTableWrap* lookup_wrap);
	//void CopyLookupTableTexture(LookupTableWrap* lookup_wrap);
	//void ReleaseScanGPUMem();
	//void CopyExtensionGPUMem(LookupTableWrap * lookup_wrap, BLAST_SequenceBlk * query);
	//void InitGPUMemforExtension(LookupTableWrap * lookup_wrap,
	//	BLAST_SequenceBlk * query);
	//void ReleaseExtenGPUMem();

	void InitGPUMemMultiSeq(const BlastSeqSrc* seq_src);
	void ReleaseGPUMemMultiSeq();
	//////////////////////////////////////////////////////////////////////////
	//extend
	//////////////////////////////////////////////////////////////////////////
	void InitGPUMemforExtension(LookupTableWrap * lookup_wrap,	BLAST_SequenceBlk * query);
 	void ReleaseExtenGPUMem();

	void SetSubjectID(int id);
	

	Int4 
		s_gpu_MBScanSubject_11_2Mod4_v1_scankernel_Opt(const LookupTableWrap* lookup_wrap,
		const BLAST_SequenceBlk* subject,
		BlastOffsetPair* NCBI_RESTRICT offset_pairs, Int4 max_hits,  
		Int4* scan_range);
	Int4 
		s_gpu_MBScanSubject_Any_scankernel_Opt(const LookupTableWrap* lookup_wrap,
		const BLAST_SequenceBlk* subject,
		BlastOffsetPair* NCBI_RESTRICT offset_pairs, Int4 max_hits,  
		Int4* scan_range);

	Int4 
		s_gpu_BlastNaExtend_Opt(BlastOffsetPair * offset_pairs, Int4 num_hits,
		const BlastInitialWordParameters * word_params,
		LookupTableWrap * lookup_wrap,
		BLAST_SequenceBlk * query,
		BLAST_SequenceBlk * subject, Int4 ** matrix,
		BlastQueryInfo * query_info,
		Blast_ExtendWord * ewp,
		BlastInitHitList * init_hitlist,
		Uint4 s_range );



#ifdef __cplusplus
}
#endif

#endif  //__GPU_BLASTN_ALL_SUBJECT_IN_ONE_H__