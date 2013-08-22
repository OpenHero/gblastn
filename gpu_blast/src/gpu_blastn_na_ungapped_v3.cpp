#include <algo/blast/core/na_ungapped.h>
#include <algo/blast/core/blast_nascan.h>
#include <algo/blast/core/blast_nalookup.h>

#include <algo/blast/gpu_blast/gpu_logfile.h> 
#include <algo/blast/gpu_blast/gpu_blastn_MB_and_smallNa.h>


void gpu_InitDBMemroy(int subject_seq_num, int max_len)
{
	InitGPUMem_DB_MultiSeq(subject_seq_num, max_len);
}
void gpu_ReleaseDBMemory()
{
	ReleaseGPUMem_DB_MultiSeq();
}

void gpu_InitQueryMemory(LookupTableWrap* lookup_wrap, BLAST_SequenceBlk* query, BlastQueryInfo* query_info)
{
	if (lookup_wrap->lut_type == eSmallNaLookupTable) {
		InitSmallQueryGPUMem(lookup_wrap, query, query_info);
	}
	else if (lookup_wrap->lut_type == eMBLookupTable) {
		InitMBQueryGPUMem(lookup_wrap, query);
	}
	else {
		BlastNaLookupTable *lookup = 
			(BlastNaLookupTable *) lookup_wrap->lut;
	}
}
void gpu_ReleaseQueryMemory(LookupTableWrap* lookup_wrap)
{
	if (lookup_wrap->lut_type == eSmallNaLookupTable) {
		ReleaseSmallQueryGPUMem();
	}
	else if (lookup_wrap->lut_type == eMBLookupTable) {
		ReleaseMBQueryGPUMem();
	}
	else {
		BlastNaLookupTable *lookup = 
			(BlastNaLookupTable *) lookup_wrap->lut;
	}
}

bool gpu_blastn_check_availability()
{
	return true;
}

void GpuLookUpInit(LookupTableWrap* lookup_wrap)
{
	if (lookup_wrap->lut_type == eSmallNaLookupTable) 
	{
		BlastSmallNaLookupTable *lookup = new BlastSmallNaLookupTable(*((BlastSmallNaLookupTable*)lookup_wrap->lut));
		lookup_wrap->lut = lookup;
	}
	else if (lookup_wrap->lut_type == eMBLookupTable) 
	{
		BlastMBLookupTable *mb_lt = new BlastMBLookupTable(*((BlastMBLookupTable*)lookup_wrap->lut));
		lookup_wrap->lut = mb_lt;
	}
}

void GpuLookUpSetUp(LookupTableWrap * lookup_wrap)
{

	if (lookup_wrap->lut_type == eSmallNaLookupTable) 
	{
		BlastSmallNaLookupTable *lookup = (BlastSmallNaLookupTable*)lookup_wrap->lut;
		Int4 scan_step = lookup->scan_step;

		ASSERT(lookup_wrap->lut_type == eSmallNaLookupTable);

		switch (lookup->lut_word_length) {
		case 8:
			lookup->scansub_callback  = (void*)s_gpu_MBScanSubject_8_1Mod4_scankernel_Opt_v3;
			break;
		} 
		lookup->extend_callback = (void*)s_gpu_BlastSmallExtend_v3;
	}
	else if (lookup_wrap->lut_type == eMBLookupTable) 
	{
		BlastMBLookupTable *mb_lt = (BlastMBLookupTable*)lookup_wrap->lut;
		ASSERT(lookup_wrap->lut_type == eMBLookupTable);

		Int4 scan_step = mb_lt->scan_step;

		switch (mb_lt->lut_word_length) {
		case 9:
			/*if (scan_step == 1)
			;
			if (scan_step == 2)
			;
			else*/
			mb_lt->scansub_callback = (void *)s_gpu_MBScanSubject_Any_scankernel_Opt_v3;
			break;

		case 10:
			/*if (scan_step == 1)
			;
			else if (scan_step == 2)
			;
			else if (scan_step == 3)
			;
			else*/
			mb_lt->scansub_callback = (void *)s_gpu_MBScanSubject_Any_scankernel_Opt_v3;
			break;

		case 11:
			switch (scan_step % COMPRESSION_RATIO) {
			case 2:
				mb_lt->scansub_callback = (void *)s_gpu_MBScanSubject_11_2Mod4_scankernel_Opt_v3;
				//mb_lt->scansub_callback = (void *)s_gpu_MBScanSubject_11_2Mod4;		 // changed by kyzhao for gpu
				break;
			}
			break;

		case 12:
			/* lookup tables of width 12 are only used
			for very large queries, and the latency of
			cache misses dominates the runtime in that
			case. Thus the extra arithmetic in the generic
			routine isn't performance-critical */
			mb_lt->scansub_callback = (void*)s_gpu_MBScanSubject_Any_scankernel_Opt_v3;
			break;
		}

		mb_lt->extend_callback = (void*)s_gpu_BlastNaExtend_Opt_v3;
	}
}

/**
* @brief Determines the scanner's offsets taking the database masking
* restrictions into account (if any). This function should be called from the
* WordFinder routines only.
*
* @param subject The subject sequence [in]
* @param word_length the real word length [in]
* @param lut_word_length the lookup table word length [in]
* @param range the structure to record seq mask index, start and
*              end of scanning pos, and start and end of current mask [in][out]
*
* @return TRUE if the scanning should proceed, FALSE otherwise
*/
static NCBI_INLINE Boolean
	s_DetermineScanningOffsets_v3(const BLAST_SequenceBlk* subject,
	Int4  word_length,
	Int4  lut_word_length,
	Int4* range)
{
	ASSERT(subject->seq_ranges);
	ASSERT(subject->num_seq_ranges >= 1);
	while (range[1] > range[2]) {
		range[0]++;
		if (range[0] >= (Int4)subject->num_seq_ranges) {
			return FALSE;
		}
		range[1] = subject->seq_ranges[range[0]].left + word_length - lut_word_length;
		range[2] = subject->seq_ranges[range[0]].right - lut_word_length;
	} 
	return TRUE;
}

/** Find all words for a given subject sequence and perform 
* ungapped extensions, assuming ordinary blastn.
* @param subject The subject sequence [in]
* @param query The query sequence (needed only for the discontiguous word 
*        case) [in]
* @param query_info concatenated query information [in]
* @param lookup_wrap Pointer to the (wrapper) lookup table structure. Only
*        traditional BLASTn lookup table supported. [in]
* @param matrix The scoring matrix [in]
* @param word_params Parameters for the initial word extension [in]
* @param ewp Structure needed for initial word information maintenance [in]
* @param offset_pairs Array for storing query and subject offsets. [in]
* @param max_hits size of offset arrays [in]
* @param init_hitlist Structure to hold all hits information. Has to be 
*        allocated up front [out]
* @param ungapped_stats Various hit counts. Not filled if NULL [out]
*/

#include <iostream>
using namespace std;

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
	BlastUngappedStats * ungapped_stats)
{
	Int4 hitsfound, total_hits = 0;
	Int4 hits_extended = 0;
	TNaScanSubjectFunction scansub = NULL;
	TNaExtendFunction extend = NULL;
	Int4 scan_range[3];
	Int4 word_length;
	Int4 lut_word_length;
	//////////////////////////////////////////////////////////////////////////
	float t_len=0;

	//cout << "Thread:" << GetCurrentThreadId() << " in finder, call adress:" << ((BlastSmallNaLookupTable *)lookup_wrap->lut)->scansub_callback << endl;

	if (lookup_wrap->lut_type == eSmallNaLookupTable) {
		BlastSmallNaLookupTable *lookup = 
			(BlastSmallNaLookupTable *) lookup_wrap->lut;
		word_length = lookup->word_length;
		lut_word_length = lookup->lut_word_length;
		scansub = (TNaScanSubjectFunction)lookup->scansub_callback;
		extend = (TNaExtendFunction)lookup->extend_callback;
	}
	else if (lookup_wrap->lut_type == eMBLookupTable) {
		BlastMBLookupTable *lookup = 
			(BlastMBLookupTable *) lookup_wrap->lut;
		if (lookup->discontiguous) {
			word_length = lookup->template_length;
			lut_word_length = lookup->template_length;
		} else {
			word_length = lookup->word_length;
			lut_word_length = lookup->lut_word_length;
		}
		scansub = (TNaScanSubjectFunction)lookup->scansub_callback;
		extend = (TNaExtendFunction)lookup->extend_callback;
	}
	else {
		BlastNaLookupTable *lookup = 
			(BlastNaLookupTable *) lookup_wrap->lut;
		word_length = lookup->word_length;
		lut_word_length = lookup->lut_word_length;
		scansub = (TNaScanSubjectFunction)lookup->scansub_callback;
		extend = (TNaExtendFunction)lookup->extend_callback;
	}

	scan_range[0] = 0;  /* subject seq mask index */
	scan_range[1] = 0;	/* start pos of scan */
	scan_range[2] = subject->length - lut_word_length; /*end pos (inclusive) of scan*/

	ASSERT(scansub);
	ASSERT(extend);
  //cout << subject->length <<endl;
	double scan_time = 0;
	double extend_time =0;
	//long scanhits = 0; 
	//long extenhits= 0;
	while(s_DetermineScanningOffsets_v3(subject, word_length, lut_word_length, scan_range)) {

		__int64 c1 = slogfile.Start();

		hitsfound = scansub(lookup_wrap, subject, offset_pairs, max_hits, &scan_range[1]);

		__int64 c2 = slogfile.End();
		slogfile.addTotalTime("Scan function time", c1, c2, false);
		slogfile.addTotalTime("scan hits", hitsfound, false);
		
		scan_time += slogfile.elaplsedTime(c1,c2);
		//scanhits += hitsfound;

		if (hitsfound == 0)
			continue;

		total_hits += hitsfound;

		c1 = slogfile.Start();

		hits_extended += extend(offset_pairs, hitsfound, word_params,
			lookup_wrap, query, subject, matrix, 
			query_info, ewp, init_hitlist, scan_range[2] + lut_word_length);

		c2 = slogfile.End();
		slogfile.addTotalTime("Extend function time",c1, c2, false);
		slogfile.addTotalNum("extended hits", hits_extended, false);

		extend_time += slogfile.elaplsedTime(c1,c2);
		//extendhits += hits_extended
	}

	cout << scan_time <<"\t" << extend_time <<"\t" << total_hits << "\t" << hits_extended<< endl;
	Blast_ExtendWordExit(ewp, subject->length);
	Blast_UngappedStatsUpdate(ungapped_stats, total_hits, hits_extended, init_hitlist->total);

	if (word_params->ungapped_extension)
		Blast_InitHitListSortByScore(init_hitlist);

	return 0;
}
