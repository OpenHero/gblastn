#ifndef __GPU_BLASTN_NA_UNGAPPED_V3_H__
#define __GPU_BLASTN_NA_UNGAPPED_V3_H__

//setup gpu
void GpuLookUpInit(LookupTableWrap* lookup_wrap);
void GpuLookUpSetUp(LookupTableWrap * lookup_wrap);

Int2 gpu_BlastNaWordFinder_v3(BLAST_SequenceBlk * subject,
	BLAST_SequenceBlk * query,
	BlastQueryInfo * query_info,
	LookupTableWrap * lookup_wrap,
	Int4 ** matrix,
	const BlastInitialWordParameters * word_params,
	Blast_ExtendWord * ewp,
	BlastOffsetPair * offset_pairs,
	Int4 max_hits,
	BlastInitHitList * init_hitlist,
	BlastUngappedStats * ungapped_stats);

void gpu_InitDBMemroy(int subject_seq_num, int max_len);
void gpu_ReleaseDBMemory();

void gpu_InitQueryMemory(LookupTableWrap* lookup_wrap, BLAST_SequenceBlk* query, BlastQueryInfo* query_info);
void gpu_ReleaseQueryMemory(LookupTableWrap* lookup_wrap);

#endif // __GPU_BLASTN_NA_UNGAPPED_H__