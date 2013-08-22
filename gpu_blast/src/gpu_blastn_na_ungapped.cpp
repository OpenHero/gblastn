#include <algo/blast/core/na_ungapped.h>
#include <algo/blast/core/blast_nascan.h>
#include <algo/blast/core/blast_nalookup.h>
#include <algo/blast/gpu_blast/gpu_blastn_GPU_setting.h>


#include <algo/blast/gpu_blast/gpu_blastn_na_ungapped.h>
#include "gpu_blastn_ungapped_extension_functions.h"
#include <algo/blast/gpu_blast/gpu_logfile.h>

#include <map>
using namespace std;


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
s_DetermineScanningOffsets(const BLAST_SequenceBlk* subject,
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
 * @return Number of hits extended. 
 */
static Int4
new_s_BlastNaExtend(const BlastOffsetPair * offset_pairs, Int4 num_hits,
                const BlastInitialWordParameters * word_params,
                LookupTableWrap * lookup_wrap,
                BLAST_SequenceBlk * query,
                BLAST_SequenceBlk * subject, Int4 ** matrix,
                BlastQueryInfo * query_info,
                Blast_ExtendWord * ewp,
                BlastInitHitList * init_hitlist,
                Uint4 s_range)
{
	//FILE *fp;
    static Int4 index2 = 0;
    Int4 index = 0;
    Int4 hits_extended = 0;
    Int4 word_length, lut_word_length, ext_to;
    BlastSeqLoc* masked_locations = NULL;
    BlastMBLookupTable *lut = (BlastMBLookupTable *) lookup_wrap->lut;
    word_length = lut->word_length;
    lut_word_length = lut->lut_word_length;
    masked_locations = lut->masked_locations;

	ext_to = word_length - lut_word_length;
    
	////*************************
 //   if(index2==0){
	//	fp = fopen("offset_pairs.txt", "w");
	//}else{
	//	fp = fopen("offset_pairs.txt", "a");
	//}
	//if(fp==NULL){
	//	printf("Cann't open the file\n");
	//	exit(-1);
	//}
 //   fprintf(fp, "index=%d ************************\n", index2++);
	////*************************
	printf("-------------------------\n");

	for (; index < num_hits; ++index) {     //Õë¶ÔÕâ¸öforÑ­»·½øÐÐ²¢ÐÐ»¯,ÄÚ²ãµÄwhileÑ­»·Ôö¼Ó²»Á¼µÄÓ°Ïì¡£
		Int4 off_index = offset_pairs[index].qs_offsets.q_off;
		Int4 q_off = lut->hashtable[off_index];

		while (q_off) {
			Int4 s_offset = offset_pairs[index].qs_offsets.s_off;
			Int4 q_offset = q_off - 1;
			/* begin with the left extension; the initialization is slightly
				faster. Point to the first base of the lookup table hit and
				work backwards */
			Int4 ext_left = 0;
			Int4 s_off = s_offset;
			Uint1 *q = query->sequence + q_offset;
			Uint1 *s = subject->sequence + s_off / COMPRESSION_RATIO;

			for (; ext_left < MIN(ext_to, s_offset); ++ext_left) {
				s_off--;
				q--;
				if (s_off % COMPRESSION_RATIO == 3)
					s--;
				if (((Uint1) (*s << (2 * (s_off % COMPRESSION_RATIO))) >> 6)
					!= *q)
					break;
			}

			/* do the right extension if the left extension did not find all
				the bases required. Begin at the first base beyond the lookup
				table hit and move forwards */

			if (ext_left < ext_to) {
				Int4 ext_right = 0;
				s_off = s_offset + lut_word_length;
				if (s_off + ext_to - ext_left > s_range){ 
					q_off = lut->next_pos[q_off];
					continue;
				}
				q = query->sequence + q_offset + lut_word_length;
				s = subject->sequence + s_off / COMPRESSION_RATIO;

				for (; ext_right < ext_to - ext_left; ++ext_right) {
					if (((Uint1) (*s << (2 * (s_off % COMPRESSION_RATIO))) >>
							6) != *q)
						break;
					s_off++;
					q++;
					if (s_off % COMPRESSION_RATIO == 0)
						s++;
				}

				/* check if enough extra matches were found */
				if (ext_left + ext_right < ext_to){
					q_off = lut->next_pos[q_off];
					continue;
				}
			}
        
			q_offset -= ext_left;
			s_offset -= ext_left;

			//fprintf(fp, "%d %d\n", q_offset, s_offset);

			hits_extended += s_BlastnExtendInitialHit(query, subject, 
												q_offset, s_offset,  
												masked_locations, 
												query_info, s_range, 
												word_length, lut_word_length,
												lookup_wrap,
												word_params, matrix,
												init_hitlist);

			q_off = lut->next_pos[q_off];
		}//end of while
	}//end of for
	//fclose(fp);
	return hits_extended;
	//return 0;
}

static Int4
s_new_withouthash_BlastNaExtend(const BlastOffsetPair * offset_pairs, Int4 num_hits,
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
    Int4 word_length, lut_word_length, ext_to;
    BlastSeqLoc* masked_locations = NULL;

    if (lookup_wrap->lut_type == eMBLookupTable) {
        BlastMBLookupTable *lut = (BlastMBLookupTable *) lookup_wrap->lut;
        word_length = lut->word_length;
        lut_word_length = lut->lut_word_length;
        masked_locations = lut->masked_locations;
    } 
    else {
        BlastNaLookupTable *lut = (BlastNaLookupTable *) lookup_wrap->lut;
        word_length = lut->word_length;
        lut_word_length = lut->lut_word_length;
        masked_locations = lut->masked_locations;
    }
    ext_to = word_length - lut_word_length;

    /* We trust that the bases of the hit itself are exact matches, 
       and look only for exact matches before and after the hit.

       Most of the time, the lookup table width is close to the word size 
       so only a few bases need examining. Also, most of the time (for
       random sequences) extensions will fail almost immediately (the
       first base examined will not match about 3/4 of the time). Thus it 
       is critical to reduce as much as possible all work that is not the 
       actual examination of sequence data */

    for (; index < num_hits; ++index) {
        Int4 s_offset = offset_pairs[index].qs_offsets.s_off;
        Int4 q_offset = offset_pairs[index].qs_offsets.q_off;

		//if (subject->oid == 20)
		//{
		//	printf("%d, %d, %d, %d\n",num_hits, index, s_offset, q_offset);
		//}

		if (s_offset > subject->length) continue;

        /* begin with the left extension; the initialization is slightly
           faster. Point to the first base of the lookup table hit and
           work backwards */

        Int4 ext_left = 0;
        Int4 s_off = s_offset;
        Uint1 *q = query->sequence + q_offset;
        Uint1 *s = subject->sequence + s_off / COMPRESSION_RATIO;

        for (; ext_left < MIN(ext_to, s_offset); ++ext_left) {
            s_off--;
            q--;
			if (q < query->sequence) break; // added by kyzhao for range limited
            if (s_off % COMPRESSION_RATIO == 3)
                s--;
            if (((Uint1) (*s << (2 * (s_off % COMPRESSION_RATIO))) >> 6)
                != *q)
                break;
        }

        /* do the right extension if the left extension did not find all
           the bases required. Begin at the first base beyond the lookup
           table hit and move forwards */

        if (ext_left < ext_to) {
            Int4 ext_right = 0;
            s_off = s_offset + lut_word_length;
            if (s_off + ext_to - ext_left > s_range) 
                continue;
            q = query->sequence + q_offset + lut_word_length;
            s = subject->sequence + s_off / COMPRESSION_RATIO;

            for (; ext_right < ext_to - ext_left; ++ext_right) {
                if (((Uint1) (*s << (2 * (s_off % COMPRESSION_RATIO))) >>
                     6) != *q)
                    break;
                s_off++;
                q++;
                if (s_off % COMPRESSION_RATIO == 0)
                    s++;
            }

            /* check if enough extra matches were found */
            if (ext_left + ext_right < ext_to)
                continue;
        }
        
        q_offset -= ext_left;
        s_offset -= ext_left;
        /* check the diagonal on which the hit lies. The boundaries
           extend from the first match of the hit to one beyond the last
           match */

        if (word_params->container_type == eDiagHash) {
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
    return hits_extended;
}


int comp_pairs (const void * elem1, const void * elem2) {
	BlastOffsetPair f = *((BlastOffsetPair*)elem1);
	BlastOffsetPair s = *((BlastOffsetPair*)elem2);
	if (f.qs_offsets.s_off > s.qs_offsets.s_off) return  1;
	if (f.qs_offsets.s_off < s.qs_offsets.s_off) return -1;
	return 0;
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

Int2 gpu_BlastNaWordFinder(BLAST_SequenceBlk * subject,
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

    if (lookup_wrap->lut_type == eSmallNaLookupTable) {
        BlastSmallNaLookupTable *lookup = 
                                (BlastSmallNaLookupTable *) lookup_wrap->lut;
        word_length = lookup->word_length;
        lut_word_length = lookup->lut_word_length;
        scansub = (TNaScanSubjectFunction)lookup->scansub_callback;
		//scansub = (TNaScanSubjectFunction)s_new_BlastSmallNaScanSubject_8_1Mod4;
        extend = (TNaExtendFunction)lookup->extend_callback;
#if GPU_RUN && GPU_SCAN_SMALL
		scansub = (TNaScanSubjectFunction)s_gpu_MBScanSubject_8_1Mod4_scankernel_Opt;
		extend = (TNaExtendFunction)s_gpu_BlastSmallExtend;
		//extend = (TNaExtendFunction)s_new_BlastSmallNaExtend;
		//extend = (TNaExtendFunction)lookup->extend_callback;
#endif
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

#if GPU_RUN
		scansub = (TNaScanSubjectFunction)s_gpu_MBScanSubject_11_2Mod4;
 		extend = (TNaExtendFunction)new_s_BlastNaExtend;

#if GPU_RUN_SCAN_v1

		scansub = (TNaScanSubjectFunction)s_gpu_MBScanSubject_11_2Mod4_v1;
#endif

#if GPU_RUN_SCAN_v1_KERNEL_v1											  
		scansub = (TNaScanSubjectFunction)s_gpu_MBScanSubject_11_2Mod4_v1_scankernel_v1;
#if MULTI_QUERIES
		scansub = (TNaScanSubjectFunction)s_gpu_MBScanSubject_11_2Mod4_v1_scankernel_Opt;
#endif
		extend = (TNaExtendFunction)s_new_withouthash_BlastNaExtend;
#endif

#if GPU_RUN_SCAN_v1_KERNEL_ANY
		if (lookup->lut_word_length == 12)
		{ 
			scansub = (TNaScanSubjectFunction)s_gpu_MBScanSubject_Any_scankernel_v1;
#if MULTI_QUERIES
			scansub = (TNaScanSubjectFunction)s_gpu_MBScanSubject_Any_scankernel_Opt;

#endif
			extend = (TNaExtendFunction)s_new_withouthash_BlastNaExtend;
		}
		
#endif
//////////////////////////////////////////////////////////////////////////
#if GPU_EXT_RUN
		extend = (TNaExtendFunction)s_gpu_BlastNaExtend;


#if GPU_EXT_RUN_V1
		extend = (TNaExtendFunction)s_gpu_BlastNaExtend_v1;
#endif

#if GPU_EXT_RUN_V2
		extend = (TNaExtendFunction)s_gpu_BlastNaExtend_v2;
#if MULTI_QUERIES
		extend = (TNaExtendFunction)s_gpu_BlastNaExtend_Opt;
#endif // MULTI_QUERIES
#endif //GPU_EXT_RUN_V2

#endif //GPU_EXT_RUN

#endif //GPU_RUN
  
#if !GPU_RUN
		scansub = (TNaScanSubjectFunction)lookup->scansub_callback;
		extend = (TNaExtendFunction)lookup->extend_callback;
#endif



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

    /* if sequence is masked, fall back to generic scanner and extender */
    //if (subject->mask_type != eNoSubjMasking) {
    //    if (lookup_wrap->lut_type == eMBLookupTable &&
    //        ((BlastMBLookupTable *) lookup_wrap->lut)->discontiguous) {
    //        /* discontiguous scan subs assumes any (non-aligned starting offset */
    //    } else {
    //        scansub = (TNaScanSubjectFunction) 
    //              BlastChooseNucleotideScanSubjectAny(lookup_wrap);
    //        if (extend != (TNaExtendFunction)s_BlastNaExtendDirect) {
    //             extend = (lookup_wrap->lut_type == eSmallNaLookupTable) 
    //                ? (TNaExtendFunction)s_BlastSmallNaExtend
    //                : (TNaExtendFunction)s_BlastNaExtend;
    //        }
    //    }
    //    /* generic scanner permits any (non-aligned) starting offset */
    //    scan_range[1] = subject->seq_ranges[0].left + word_length - lut_word_length;
    //    scan_range[2] = subject->seq_ranges[0].right - lut_word_length;
    //}

    ASSERT(scansub);
    ASSERT(extend);

   int ixxxxx = 1; 
	while(s_DetermineScanningOffsets(subject, word_length, lut_word_length, scan_range)) {

	__int64 c1 = slogfile.Start();

	hitsfound = scansub(lookup_wrap, subject, offset_pairs, max_hits, &scan_range[1]);

	__int64 c2 = slogfile.End();
	slogfile.addTotalTime("Scan function time", c1, c2, false);
	slogfile.addTotalTime("scan hits", hitsfound, false);

		if (hitsfound == 0)
			continue;

		total_hits += hitsfound;

		//qsort(offset_pairs,total_hits,sizeof(BlastOffsetPair), comp_pairs);
		//for (int i = 0; i < total_hits; i++)
		//{
		//	printf("%d:%d\n",offset_pairs[i].qs_offsets.q_off,offset_pairs[i].qs_offsets.s_off);
		//}
		//printf("------------------------%d-------------\n", ixxxxx++);

		c1 = slogfile.Start();

		hits_extended += extend(offset_pairs, hitsfound, word_params,
			lookup_wrap, query, subject, matrix, 
			query_info, ewp, init_hitlist, scan_range[2] + lut_word_length);
		//printf("----------------after--------------\n");
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


int search_binary(vector<int> offset_array, int key)   
{  
	int left = 0;  
	int right = offset_array.size() -1;  

	int middle = 0;

	while (left <= right)  
	{  
		middle = (left + right)/2;
		int temp = offset_array[middle];
		if (temp == key)  
		{  
			return middle;  
		}  

		else if (temp > key)  
		{  
			right = middle - 1;  
		}  

		else  
			left = middle + 1;  

	}  

	return left-1;  

}  

int oids = 0; 

int BlastnGPUOneVolOffset(BlastHSPList& hsp_list, vector<int> offset_array, vector<BlastHSPList*>& hsp_list_out)
{

	int num = hsp_list.hspcnt; 

	int j = 0;

	//for (int i = 0; i < num; i++)
	//{
	//	BlastSeg* temp_Seg = &((hsp_list.hsp_array[i])->subject);
	//   	int position = search_binary(offset_array, temp_Seg->offset);
	//	hsp_list.oid = oids + position;
	//	int vol_offset = offset_array[position];
	//	temp_Seg->offset = temp_Seg->offset - vol_offset;
	//	temp_Seg->end = temp_Seg->end - vol_offset;
	//	temp_Seg->gapped_start = temp_Seg->gapped_start - vol_offset;
	//}
	//oids += offset_array.size();

	map<int, vector<BlastHSP*>> t_hsp_map;
	map<int, vector<BlastHSP*>>::iterator itr;
	 typedef pair <int, vector<BlastHSP*>> hsp_Pair;
	 

	int position = 0;
	for (int i = 0; i < num; i++)
	{

		BlastHSP* t_hsp = hsp_list.hsp_array[i];


		BlastSeg* temp_Seg = &(t_hsp->subject);
		position = search_binary(offset_array, temp_Seg->offset);

		int vol_offset = offset_array[position];
		temp_Seg->offset = temp_Seg->offset - vol_offset;
		temp_Seg->end = temp_Seg->end - vol_offset;
		temp_Seg->gapped_start = temp_Seg->gapped_start - vol_offset;

		int new_oid = position + oids;
		hsp_list.oid = new_oid;

		
		itr = t_hsp_map.find(new_oid);
		if (itr == t_hsp_map.end())
		{
			vector<BlastHSP*> t_v_hsp;
			t_v_hsp.push_back(t_hsp);
			t_hsp_map.insert(hsp_Pair(new_oid, t_v_hsp));
		}else
		{
			itr->second.push_back(t_hsp);
		}
	}
	oids += offset_array.size();
	//x++;

	int hsp_size = t_hsp_map.size();
	//
	itr = t_hsp_map.begin();
	//int x =0; 
	while (itr != t_hsp_map.end())
	{
		BlastHSPList* t_hsp_list = (BlastHSPList*)	calloc(1, sizeof(BlastHSPList));
		t_hsp_list->oid = itr->first;
		t_hsp_list->query_index = hsp_list.query_index;
		t_hsp_list->hspcnt = itr->second.size();
		t_hsp_list->allocated = hsp_list.allocated;
		t_hsp_list->hsp_max = hsp_list.hsp_max;
		t_hsp_list->do_not_reallocate = hsp_list.do_not_reallocate;
		t_hsp_list->best_evalue = hsp_list.best_evalue;



		BlastHSP** t_hsp_tt = (BlastHSP**) calloc(t_hsp_list->hspcnt, sizeof(BlastHSP*));
		

		for(int i = 0; i < t_hsp_list->hspcnt; i++)
		{
			t_hsp_tt[i] = itr->second[i];
		}
		t_hsp_list->hsp_array = t_hsp_tt;

		hsp_list_out.push_back(t_hsp_list);

		itr++;
	}


	return hsp_size;
}