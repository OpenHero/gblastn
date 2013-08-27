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

static Int4 s_BlastSmallNaScanSubject_8_4(const LookupTableWrap * lookup_wrap,
	const BLAST_SequenceBlk * subject,
	BlastOffsetPair * NCBI_RESTRICT offset_pairs,
	Int4 max_hits, Int4 * scan_range)
{
	BlastSmallNaLookupTable *lookup = 
		(BlastSmallNaLookupTable *) lookup_wrap->lut;
	const Int4 kLutWordLength = 8;
	const Int4 kLutWordMask = (1 << (2 * kLutWordLength)) - 1;
	Int4 num_words = (scan_range[1] - scan_range[0]) / 4 + 1;
	Uint1 *s = subject->sequence + scan_range[0] / COMPRESSION_RATIO;
	Int4 total_hits = 0;
	Int2 *backbone = lookup->final_backbone;
	Int2 *overflow = lookup->overflow;
	Int4 init_index;
	Int4 index;

	ASSERT(lookup_wrap->lut_type == eSmallNaLookupTable);
	ASSERT(lookup->lut_word_length == 8);
	ASSERT(lookup->scan_step == 4);
	max_hits -= lookup->longest_chain;

	init_index = s[0];
	switch (num_words % 8) {
	case 1: s -= 7; scan_range[0] -= 28; goto byte_7;
	case 2: s -= 6; scan_range[0] -= 24; goto byte_6;
	case 3: s -= 5; scan_range[0] -= 20; goto byte_5;
	case 4: s -= 4; scan_range[0] -= 16; goto byte_4;
	case 5: s -= 3; scan_range[0] -= 12; goto byte_3;
	case 6: s -= 2; scan_range[0] -=  8; goto byte_2;
	case 7: s -= 1; scan_range[0] -=  4; goto byte_1;
	}

	while (scan_range[0] <= scan_range[1]) {

		init_index = init_index << 8 | s[1];
		index = backbone[init_index & kLutWordMask];
		SMALL_NA_ACCESS_HITS(0);
byte_1:
		init_index = init_index << 8 | s[2];
		index = backbone[init_index & kLutWordMask];
		SMALL_NA_ACCESS_HITS(4);
byte_2:
		init_index = init_index << 8 | s[3];
		index = backbone[init_index & kLutWordMask];
		SMALL_NA_ACCESS_HITS(8);
byte_3:
		init_index = init_index << 8 | s[4];
		index = backbone[init_index & kLutWordMask];
		SMALL_NA_ACCESS_HITS(12);
byte_4:
		init_index = init_index << 8 | s[5];
		index = backbone[init_index & kLutWordMask];
		SMALL_NA_ACCESS_HITS(16);
byte_5:
		init_index = init_index << 8 | s[6];
		index = backbone[init_index & kLutWordMask];
		SMALL_NA_ACCESS_HITS(20);
byte_6:
		init_index = init_index << 8 | s[7];
		index = backbone[init_index & kLutWordMask];
		SMALL_NA_ACCESS_HITS(24);
byte_7:
		init_index = init_index << 8 | s[8];
		s += 8;
		index = backbone[init_index & kLutWordMask];
		SMALL_NA_ACCESS_HITS(28);
		scan_range[0] += 32;
	}

	return total_hits;
}

static Int4
	s_BlastNaExtendDirect(BlastOffsetPair * offset_pairs, Int4 num_hits,
	const BlastInitialWordParameters * word_params,
	LookupTableWrap * lookup_wrap,
	BLAST_SequenceBlk * query,
	BLAST_SequenceBlk * subject, Int4 ** matrix,
	BlastQueryInfo * query_info,
	Blast_ExtendWord * ewp,
	BlastInitHitList * init_hitlist,
	Uint4 s_range)
{
	return 0;
}

static Int4
	s_BlastSmallNaExtendAlignedOneByte(const BlastOffsetPair * offset_pairs, 
	Int4 num_hits,
	const BlastInitialWordParameters * word_params,
	LookupTableWrap * lookup_wrap,
	BLAST_SequenceBlk * query, BLAST_SequenceBlk * subject,
	Int4 ** matrix, BlastQueryInfo * query_info,
	Blast_ExtendWord * ewp,
	BlastInitHitList * init_hitlist,
	Uint4 s_range)
{
	Int4 index = 0;
    Int4 hits_extended = 0;
    BlastSmallNaLookupTable *lut = (BlastSmallNaLookupTable *) lookup_wrap->lut;
    Int4 word_length = lut->word_length;
    Int4 lut_word_length = lut->lut_word_length;
    Int4 ext_to = word_length - lut_word_length;
    Uint1 *q = query->compressed_nuc_seq;
    Uint1 *s = subject->sequence;

#if 0
for (; index < num_hits; ++index) {
        Int4 s_offset = offset_pairs[index].qs_offsets.s_off;
        Int4 q_offset = offset_pairs[index].qs_offsets.q_off;
        Int4 ext_left = 0;

        Int4 context = BSearchContextInfo(q_offset, query_info);
        Int4 q_start = query_info->contexts[context].query_offset;
        Int4 q_range = q_start + query_info->contexts[context].query_length;

        /* the seed is assumed to start on a multiple of 4 bases
           in the subject sequence. Look for up to 4 exact matches
           to the left. The index into q[] below can
           technically be negative, but the compressed version
           of the query has extra pad bytes before q[0] */

        if ( (s_offset > 0) && (q_offset > 0) ) {
            Uint1 q_byte = q[q_offset - 4];
            Uint1 s_byte = s[s_offset / COMPRESSION_RATIO - 1];
            ext_left = s_ExactMatchExtendLeft[q_byte ^ s_byte];
            ext_left = MIN(MIN(ext_left, ext_to), q_offset - q_start);
        }

        /* look for up to 4 exact matches to the right of the seed */

        if ((ext_left < ext_to) && ((q_offset + lut_word_length) < query->length)) {
            Uint1 q_byte = q[q_offset + lut_word_length];
            Uint1 s_byte = s[(s_offset + lut_word_length) / COMPRESSION_RATIO];
            Int4 ext_right = s_ExactMatchExtendRight[q_byte ^ s_byte];
            ext_right = MIN(MIN(ext_right, s_range - (s_offset + lut_word_length)), 
                                           q_range - (q_offset + lut_word_length));
            if (ext_left + ext_right < ext_to)
                continue;
        }

        q_offset -= ext_left;
        s_offset -= ext_left;
        
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
        }
        else {
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
#endif
    return hits_extended;
}

static Int4 s_MBScanSubject_10_1(const LookupTableWrap* lookup_wrap,
	const BLAST_SequenceBlk* subject,
	BlastOffsetPair* NCBI_RESTRICT offset_pairs, Int4 max_hits,  
	Int4* scan_range)
{
	return 0;
}

static Int4 s_MBScanSubject_11_1Mod4(const LookupTableWrap* lookup_wrap,
	const BLAST_SequenceBlk* subject,
	BlastOffsetPair* NCBI_RESTRICT offset_pairs, Int4 max_hits,  
	Int4* scan_range)
{
return 0;
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
			if (scan_step == 4) {
				lookup->scansub_callback = (void *)s_BlastSmallNaScanSubject_8_4;
			}else
			{
				lookup->scansub_callback  = (void*)s_gpu_MBScanSubject_8_1Mod4_scankernel_Opt_v3;
			}
			break;
		}
		//lookup_wrap->lookup_callback = (void *)s_SmallNaLookup;
		if (lookup->lut_word_length == lookup->word_length)
			lookup->extend_callback = (void *)s_BlastNaExtendDirect;
		else if (lookup->lut_word_length % COMPRESSION_RATIO == 0 &&
			lookup->scan_step % COMPRESSION_RATIO == 0 &&
			lookup->word_length - lookup->lut_word_length <= 4)
			lookup->extend_callback = (void *)s_BlastSmallNaExtendAlignedOneByte;
		else
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
			if (scan_step == 1)
				mb_lt->scansub_callback = (void *)s_MBScanSubject_10_1;
			else
				mb_lt->scansub_callback = (void *)s_gpu_MBScanSubject_Any_scankernel_Opt_v3;
			break;

		case 11:
			switch (scan_step % COMPRESSION_RATIO) {
			case 1:
				//mb_lt->scansub_callback = (void *)s_MBScanSubject_11_1Mod4;
				mb_lt->scansub_callback = (void *)s_gpu_MBScanSubject_11_1Mod4_scankernel_Opt_v3;
				break;
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

		if (mb_lt->lut_word_length == mb_lt->word_length || mb_lt->discontiguous)
			mb_lt->extend_callback = (void *)s_new_BlastNaExtendDirect;
			//mb_lt->extend_callback = (void*)s_gpu_BlastNaExtend_Opt_v3;
		else
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

	while(s_DetermineScanningOffsets_v3(subject, word_length, lut_word_length, scan_range)) {

		__int64 c1 = slogfile.Start();

		hitsfound = scansub(lookup_wrap, subject, offset_pairs, max_hits, &scan_range[1]);

		__int64 c2 = slogfile.End();
		slogfile.addTotalTime("Scan function time", c1, c2, false);
		slogfile.addTotalTime("scan hits", hitsfound, false);

		if (hitsfound == 0)
			continue;

		total_hits += hitsfound;

		c1 = slogfile.Start();

		hits_extended += extend(offset_pairs, hitsfound, word_params,
			lookup_wrap, query, subject, matrix, 
			query_info, ewp, init_hitlist, scan_range[2] + lut_word_length);

		c2 = slogfile.End();
		slogfile.addTotalTime("Extend function time",c1, c2 ,false);
		slogfile.addTotalNum("extended hits", hits_extended, false);
	}
	Blast_ExtendWordExit(ewp, subject->length);
	Blast_UngappedStatsUpdate(ungapped_stats, total_hits, hits_extended, init_hitlist->total);

	if (word_params->ungapped_extension)
		Blast_InitHitListSortByScore(init_hitlist);

	return 0;
}
