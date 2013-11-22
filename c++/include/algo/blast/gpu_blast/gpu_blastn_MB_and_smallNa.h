#ifndef __GPU_BLASTN_NA_AND_SMALL_H__
#define __GPU_BLASTN_NA_AND_SMALL_H__

#include <algo/blast/gpu_blast/gpu_blastn_config.hpp>
#include <algo/blast/gpu_blast/gpu_blastn.h>

void InitGPUMem_DB_MultiSeq(int subject_seq_num, int max_len);
void ReleaseGPUMem_DB_MultiSeq();

//////////////////////////////////////////////////////////////////////////
//small
void InitSmallQueryGPUMem(LookupTableWrap* lookup_wrap, BLAST_SequenceBlk* query, BlastQueryInfo* query_info);
void ReleaseSmallQueryGPUMem();

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
	s_gpu_MBScanSubject_8_1Mod4_scankernel_Opt_v3(const LookupTableWrap* lookup_wrap,
	const BLAST_SequenceBlk* subject,
	BlastOffsetPair* NCBI_RESTRICT offset_pairs, Int4 max_hits,  
	Int4* scan_range);
Int4 
	s_gpu_BlastSmallExtend_v3(BlastOffsetPair * offset_pairs, Int4 num_hits,
	const BlastInitialWordParameters * word_params,
	LookupTableWrap * lookup_wrap,
	BLAST_SequenceBlk * query,
	BLAST_SequenceBlk * subject, Int4 ** matrix,
	BlastQueryInfo * query_info,
	Blast_ExtendWord * ewp,
	BlastInitHitList * init_hitlist,
	Uint4 s_range );

Int4 
	s_gpu_BlastSmallNaScanSubject_8_4(const LookupTableWrap* lookup_wrap,
	const BLAST_SequenceBlk* subject,
	BlastOffsetPair* NCBI_RESTRICT offset_pairs, Int4 max_hits,  
	Int4* scan_range);
Int4 
	s_gpu_BlastSmallNaExtendAlignedOneByte(BlastOffsetPair * offset_pairs, Int4 num_hits,
	const BlastInitialWordParameters * word_params,
	LookupTableWrap * lookup_wrap,
	BLAST_SequenceBlk * query,
	BLAST_SequenceBlk * subject, Int4 ** matrix,
	BlastQueryInfo * query_info,
	Blast_ExtendWord * ewp,
	BlastInitHitList * init_hitlist,
	Uint4 s_range );
//////////////////////////////////////////////////////////////////////////
//medium and large 
void InitMBQueryGPUMem(LookupTableWrap * lookup_wrap,	BLAST_SequenceBlk * query);
void ReleaseMBQueryGPUMem();

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
	s_gpu_MBScanSubject_11_2Mod4_scankernel_Opt_v3(const LookupTableWrap* lookup_wrap,
	const BLAST_SequenceBlk* subject,
	BlastOffsetPair* NCBI_RESTRICT offset_pairs, Int4 max_hits,  
	Int4* scan_range);
Int4 
	s_gpu_MBScanSubject_11_1Mod4_scankernel_Opt_v3(const LookupTableWrap* lookup_wrap,
	const BLAST_SequenceBlk* subject,
	BlastOffsetPair* NCBI_RESTRICT offset_pairs, Int4 max_hits,  
	Int4* scan_range);
Int4 
	s_gpu_MBScanSubject_Any_scankernel_Opt_v3(const LookupTableWrap* lookup_wrap,
	const BLAST_SequenceBlk* subject,
	BlastOffsetPair* NCBI_RESTRICT offset_pairs, Int4 max_hits,  
	Int4* scan_range);
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
	Uint4 s_range );

Int4
	s_new_BlastNaExtendDirect(BlastOffsetPair * offset_pairs, Int4 num_hits,
	const BlastInitialWordParameters * word_params,
	LookupTableWrap * lookup_wrap,
	BLAST_SequenceBlk * query,
	BLAST_SequenceBlk * subject, Int4 ** matrix,
	BlastQueryInfo * query_info,
	Blast_ExtendWord * ewp,
	BlastInitHitList * init_hitlist,
	Uint4 s_range);

#endif