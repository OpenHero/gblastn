//#ifndef __UNGPAPPED_EXTENSION_FUNCTIONS_H__
//#define __UNGPAPPED_EXTENSION_FUNCTIONS_H__

//#if _MSC_VER > 1000
//#pragma once
//#endif // _MSC_VER > 1000

/*
 * include file used by s_DetermineNaScanningOffsets
 */
#include <algo/blast/core/na_ungapped.h>
#include <algo/blast/core/blast_nalookup.h>
#include <algo/blast/core/blast_nascan.h>
#include <algo/blast/core/mb_indexed_lookup.h>
#include <algo/blast/core/blast_util.h> /* for NCBI2NA_UNPACK_BASE macros */
#include <algo/blast/core/blast_gapalign.h>

/************************************************************************/
/* Perform ungapped extension                                           */
/************************************************************************/

/** Perform ungapped extension of a word hit, using a score
 *  matrix and extending one base at a time
 * @param query The query sequence [in]
 * @param subject The subject sequence [in]
 * @param matrix The scoring matrix [in]
 * @param q_off The offset of a word in query [in]
 * @param s_off The offset of a word in subject [in]
 * @param X The drop-off parameter for the ungapped extension [in]
 * @param ungapped_data The ungapped extension information [out]
 */
static void
s_NuclUngappedExtendExact(BLAST_SequenceBlk * query,
                          BLAST_SequenceBlk * subject, Int4 ** matrix,
                          Int4 q_off, Int4 s_off, Int4 X,
                          BlastUngappedData * ungapped_data)
{
    Uint1 *q;
    Int4 sum, score;
    Uint1 ch;
    Uint1 *subject0, *sf, *q_beg, *q_end, *s, *start;
    Int2 remainder, base;
    Int4 q_avail, s_avail;

    base = 3 - (s_off % 4);

    subject0 = subject->sequence;
    q_avail = query->length - q_off;
    s_avail = subject->length - s_off;

    q = q_beg = q_end = query->sequence + q_off;
    s = subject0 + s_off / COMPRESSION_RATIO;
    if (q_off < s_off) {
        start = subject0 + (s_off - q_off) / COMPRESSION_RATIO;
        remainder = 3 - ((s_off - q_off) % COMPRESSION_RATIO);
    } else {
        start = subject0;
        remainder = 3;
    }

    score = 0;
    sum = 0;

/*
   There is a trick in the loop below that you can't see from the code.
   X is negative and when sum becomes more negative than X we break out of
   the loop.  The reason that X is guaranteed to become negative is because
   there is a sentinel at the beginning of the query sequence, so if you hit
   that you get a big negative value.
*/

    /* extend to the left */
    while ((s > start) || (s == start && base < remainder)) {
        if (base == 3) {
            s--;
            base = 0;
        } else {
            ++base;
        }
        ch = *s;
        if ((sum += matrix[*--q][NCBI2NA_UNPACK_BASE(ch, base)]) > 0) {
            q_beg = q;
            score += sum;
            sum = 0;
        } else if (sum < X) {
            break;
        }
    }

    ungapped_data->q_start = q_beg - query->sequence;
    ungapped_data->s_start = s_off - (q_off - ungapped_data->q_start);

    if (q_avail < s_avail) {
        sf = subject0 + (s_off + q_avail) / COMPRESSION_RATIO;
        remainder = 3 - ((s_off + q_avail) % COMPRESSION_RATIO);
    } else {
        sf = subject0 + (subject->length) / COMPRESSION_RATIO;
        remainder = 3 - ((subject->length) % COMPRESSION_RATIO);
    }

    /* extend to the right */
    q = query->sequence + q_off;
    s = subject0 + s_off / COMPRESSION_RATIO;
    sum = 0;
    base = 3 - (s_off % COMPRESSION_RATIO);

    while (s < sf || (s == sf && base > remainder)) {
        ch = *s;
        if ((sum += matrix[*q++][NCBI2NA_UNPACK_BASE(ch, base)]) > 0) {
            q_end = q;
            score += sum;
            sum = 0;
        } else if (sum < X)
            break;
        if (base == 0) {
            base = 3;
            s++;
        } else
            base--;
    }

    ungapped_data->length = q_end - q_beg;
    ungapped_data->score = score;
}

/** Perform ungapped extension of a word hit. Use an approximate method
 * and revert to rigorous ungapped alignment if the approximate score
 * is high enough
 * @param query The query sequence [in]
 * @param subject The subject sequence [in]
 * @param matrix The scoring matrix [in]
 * @param q_off The offset of a word in query [in]
 * @param s_match_end The first offset in the subject sequence that 
 *              is not known to exactly match the query sequence [in]
 * @param s_off The offset of a word in subject [in]
 * @param X The drop-off parameter for the ungapped extension [in]
 * @param ungapped_data The ungapped extension information [out]
 * @param score_table Array containing precompute sums of
 *                    match and mismatch scores [in]
 * @param reduced_cutoff Score beyond which a rigorous extension is used [in]
 */
static void
s_NuclUngappedExtend(BLAST_SequenceBlk * query,
                     BLAST_SequenceBlk * subject, Int4 ** matrix,
                     Int4 q_off, Int4 s_match_end, Int4 s_off,
                     Int4 X, BlastUngappedData * ungapped_data,
                     const Int4 * score_table, Int4 reduced_cutoff)
{
    Uint1 *q_start = query->sequence;
    Uint1 *s_start = subject->sequence;
    Uint1 *q;
    Uint1 *s;
    Int4 sum, score;
    Int4 i, len;
    Uint1 *new_q;
    Int4 q_ext, s_ext;

    /* The left extension begins behind (q_ext,s_ext); this is the first
       4-base boundary after s_off. */
    len = (COMPRESSION_RATIO - (s_off % COMPRESSION_RATIO)) %
        COMPRESSION_RATIO;
    q_ext = q_off + len;
    s_ext = s_off + len;
    q = q_start + q_ext;
    s = s_start + s_ext / COMPRESSION_RATIO;
    len = MIN(q_ext, s_ext) / COMPRESSION_RATIO;

    score = 0;
    sum = 0;
    new_q = q;

    for (i = 0; i < len; s--, q -= 4, i++) {
        Uint1 s_byte = s[-1];
        Uint1 q_byte = (q[-4] << 6) | (q[-3] << 4) | (q[-2] << 2) | q[-1];

        sum += score_table[q_byte ^ s_byte];
        if (sum > 0) {
            new_q = q - 4;
            score += sum;
            sum = 0;
        }
        if (sum < X) {
            break;
        }
    }

    /* record the start point of the extension */

    ungapped_data->q_start = new_q - q_start;
    ungapped_data->s_start = s_ext - (q_ext - ungapped_data->q_start);

    /* the right extension begins at the first bases not examined by the
       previous loop */

    q = q_start + q_ext;
    s = s_start + s_ext / COMPRESSION_RATIO;
    len = MIN(query->length - q_ext, subject->length - s_ext) /
        COMPRESSION_RATIO;
    sum = 0;
    new_q = q;

    for (i = 0; i < len; s++, q += 4, i++) {
        Uint1 s_byte = s[0];
        Uint1 q_byte = (q[0] << 6) | (q[1] << 4) | (q[2] << 2) | q[3];

        sum += score_table[q_byte ^ s_byte];
        if (sum > 0) {
            new_q = q + 3;
            score += sum;
            sum = 0;
        }
        if (sum < X) {
            break;
        }
    }

    if (score >= reduced_cutoff) {
        /* the current score is high enough; throw away the alignment and
           recompute it rigorously */
        s_NuclUngappedExtendExact(query, subject, matrix, q_off,
                                  s_off, X, ungapped_data);
    } else {
        /* record the length and score of the extension. Make sure the
           alignment extends at least to s_match_end */
        ungapped_data->score = score;
        ungapped_data->length = MAX(s_match_end - ungapped_data->s_start,
                                    (new_q - q_start) -
                                    ungapped_data->q_start + 1);
    }
}
/** Test to see if seed->q_off exists in lookup table
 * @param lookup_wrap The lookup table wrap structure [in]
 * @param subject Subject sequence data [in]
 * @s_off The starting offset of the seed [in]
 * @lut_word_length The length of the lookup word [in]
 */
static NCBI_INLINE Boolean
s_IsSeedMasked(const LookupTableWrap   * lookup_wrap,
               const BLAST_SequenceBlk * subject,
               Int4                      s_off,
               Int4                      lut_word_length,
               Int4                      q_pos)
{
    Uint1 *s  = subject->sequence + s_off / COMPRESSION_RATIO;
    Int4 shift = 2* (16 - s_off % COMPRESSION_RATIO - lut_word_length);
    Int4 index = (s[0] << 24 | s[1] << 16 | s[2] << 8 | s[3]) >> shift;
    return !(((T_Lookup_Callback)(lookup_wrap->lookup_callback))
                                         (lookup_wrap, index, q_pos));
}
/** Check the mini-extended word against masked query regions.
 * @param query Query sequence data [in]
 * @param subject Subject sequence data [in]
 * @param q_off Query offset [in][out]
 * @param s_off Subject offset [in][out]
 * @param locations of the masked query regions [in]
 * @param query_info of the masked query regions [in]
 * @param s_range the open bound of subject region [in]
 * @param word_length length of word [in]
 * @param lut_word_length length of lookup table word [in]
 * @param extended if successful, the actual bases extended [out]
 * @return 0,1 for non-word, single word
 */
static Int4
s_CheckMask(BLAST_SequenceBlk * query,
             BLAST_SequenceBlk * subject,
             Int4 *q_off, Int4 *s_off,
             BlastSeqLoc* locations, 
             BlastQueryInfo * query_info,
             Uint4 s_range,
             Uint4 word_length,
             Uint4 lut_word_length,
             const LookupTableWrap* lookup_wrap,
             Int4 * extended)
{
    Int4 context, q_range;
    Int4 ext_to, ext_max;
    Int4 q_end = *q_off + word_length;
    Int4 s_end = *s_off + word_length;
    Int4 s_pos, q_pos;

    *extended = 0;

    /* No need to check if mini-extension is not performed.*/
    if (word_length == lut_word_length) return 1;

    /* Find the query context boundary */
    context = BSearchContextInfo(q_end, query_info);
    q_range = query_info->contexts[context].query_offset 
            + query_info->contexts[context].query_length;

    /* check query mask at two ends */
    if (locations) {
        /* check for right end first */
        if (s_IsSeedMasked(lookup_wrap, subject, 
                           s_end - lut_word_length, 
                           lut_word_length, 
                           q_end - lut_word_length)) return 0;

        /* search for valid left end and reposition q_off */
        for (; TRUE; ++(*s_off), ++(*q_off)) {
            if (!s_IsSeedMasked(lookup_wrap, subject,
                                *s_off, lut_word_length, *q_off)) break;
        }
    }

    ext_to = word_length - (q_end - (*q_off));
    ext_max = MIN(q_range - q_end, s_range - s_end);

    /* shift to the right, and check query mask inside */
    if (ext_to || locations) {

        if (ext_to > ext_max) return 0;
        q_end += ext_to;
        s_end += ext_to;

        for (s_pos = s_end - lut_word_length, 
             q_pos = q_end - lut_word_length;
             s_pos > *s_off; 
             s_pos -= lut_word_length,
             q_pos -= lut_word_length) {
             if (s_IsSeedMasked(lookup_wrap, subject,
                           s_pos, lut_word_length, q_pos)) return 0;
        }

        (*extended) = ext_to;
    }
    /* if we get here, single word check is passed */
    return 1;
}

/** Perform ungapped extension given an offset pair, and save the initial 
 * hit information if the hit qualifies. This function assumes that the
 * exact match has already been extended to the word size parameter. 
 * @param query The query sequence [in]
 * @param subject The subject sequence [in]
 * @param q_off The offset in the query sequence [in]
 * @param s_off The offset in the subject sequence [in]
 * @param query_mask Structure containing query mask ranges [in]
 * @param query_info Structure containing query context ranges [in]
 * @param s_range Subject range [in]
 * @param word_length The length of the hit [in]
 * @param word_lut_length The length of the lookup table word [in]
 * @param word_params The parameters related to initial word extension [in]
 * @param matrix the substitution matrix for ungapped extension [in]
 * @param init_hitlist The structure containing information about all 
 *                     initial hits [in] [out]
 * @return 1 if hit was extended, 0 if not
 */

static Int4
s_BlastnExtendInitialHit(BLAST_SequenceBlk * query,
                        BLAST_SequenceBlk * subject, 
                        Int4 q_off, Int4 s_off, 
                        BlastSeqLoc* query_mask,
                        BlastQueryInfo * query_info,
                        Int4 s_range,
                        Int4 word_length, Int4 lut_word_length,
                        const LookupTableWrap * lut,
                        const BlastInitialWordParameters * word_params,
                        Int4 ** matrix, 
                        BlastInitHitList * init_hitlist)
{
    Int4 s_end;
    Int4 extended = 0;
    BlastUngappedData *ungapped_data;
    BlastUngappedData dummy_ungapped_data;
    Int4 hit_ready = 1;
    BlastUngappedCutoffs *cutoffs = NULL;

    s_end = s_off + word_length;

    /* check the masks for the word */
    if(!s_CheckMask(query, subject, &q_off, &s_off,
                      query_mask, query_info, s_range, 
                      word_length, lut_word_length, lut, &extended)) return 0;

    /* update the right end*/
    s_end += extended;
       
    if (hit_ready) {
        if (word_params->ungapped_extension) {
            Int4 context = BSearchContextInfo(q_off, query_info);
            cutoffs = word_params->cutoffs + context;
            ungapped_data = &dummy_ungapped_data;

            /* 
             * Skip use of the scoring table and go straight to the matrix
             * based extension if matrix_only_scoring is set.  Used by
             * app rmblastn.
             * -RMH-
             */
            if ( word_params->options->program_number == eBlastTypeBlastn &&
                 word_params->matrix_only_scoring )
            {
               s_NuclUngappedExtendExact(query, subject, matrix, q_off,
                                  s_off, -(cutoffs->x_dropoff), ungapped_data);
            }else {
               s_NuclUngappedExtend(query, subject, matrix, q_off, s_end, s_off,
                                 -(cutoffs->x_dropoff), ungapped_data,
                                 word_params->nucl_score_table,
                                 cutoffs->reduced_nucl_cutoff_score);
            }

            if (ungapped_data->score >= cutoffs->cutoff_score) {
                BlastUngappedData *final_data =
                    (BlastUngappedData *) malloc(sizeof(BlastUngappedData));
                *final_data = *ungapped_data;
                BLAST_SaveInitialHit(init_hitlist, q_off, s_off, final_data);
            } else {
                hit_ready = 0;
            }
        } else {
            ungapped_data = NULL;
            BLAST_SaveInitialHit(init_hitlist, q_off, s_off, ungapped_data);
        }
    } 

    return hit_ready;
}
//////////////////////////////////////////////////////////////////////////


/**
 * Attempt to retrieve information associated with diagonal diag.
 * @param table The hash table [in]
 * @param diag The diagonal to be retrieved [in]
 * @param level The offset of the last hit on the specified diagonal [out]
 * @param hit_saved Whether or not the last hit on the specified diagonal was saved [out]
 * @param hit_length length of the last hit on the specified diagonal [out]
 * @return 1 if successful, 0 if no hit was found on the specified diagonal.
 */
static NCBI_INLINE Int4 s_BlastDiagHashRetrieve(BLAST_DiagHash * table,
                                                Int4 diag, Int4 * level,
                                                Int4 * hit_len,
                                                Int4 * hit_saved)
{
    /* see http://lxr.linux.no/source/include/linux/hash.h */
    /* mod operator will be strength-reduced to an and by the compiler */
    Uint4 bucket = ((Uint4) diag * 0x9E370001) % DIAGHASH_NUM_BUCKETS;
    Uint4 index = table->backbone[bucket];

    while (index) {
        if (table->chain[index].diag == diag) {
            *level = table->chain[index].level;
            *hit_len = table->chain[index].hit_len;
            *hit_saved = table->chain[index].hit_saved;
            return 1;
        }

        index = table->chain[index].next;
    }
    return 0;
}
/**
 * Attempt to store information associated with diagonal diag.
 * Cleans up defunct entries along the way or allocates more space if necessary.
 * @param table The hash table [in]
 * @param diag The diagonal to be stored [in]
 * @param level The offset of the hit to be stored [in]
 * @param len The length of the hit to be stored [in]
 * @param hit_saved Whether or not this hit was stored [in]
 * @param s_end Needed to clean up defunct entries [in]
 * @param window_size Needed to clean up defunct entries [in]
 * @return 1 if successful, 0 if memory allocation failed.
 */
static NCBI_INLINE Int4 s_BlastDiagHashInsert(BLAST_DiagHash * table,
                                              Int4 diag, Int4 level,
                                              Int4 len,
                                              Int4 hit_saved,
                                              Int4 s_off,
                                              Int4 window_size)
{
    Uint4 bucket = ((Uint4) diag * 0x9E370001) % DIAGHASH_NUM_BUCKETS;
    Uint4 index = table->backbone[bucket];
    DiagHashCell *cell = NULL;

    while (index) {
        /* if we find what we're looking for, save into it */
        if (table->chain[index].diag == diag) {
            table->chain[index].level = level;
            table->chain[index].hit_len = len;
            table->chain[index].hit_saved = hit_saved;
            return 1;
        }
        /* otherwise, if this hit is stale, save into it. */
        else {
            /* if this hit is stale, save into it. */
            if ( s_off - table->chain[index].level > window_size) {
                table->chain[index].diag = diag;
                table->chain[index].level = level;
                table->chain[index].hit_len = len;
                table->chain[index].hit_saved = hit_saved;
                return 1;
            }
        }
        index = table->chain[index].next;
    }

    /* if we got this far, we were unable to replace any existing entries. */

    /* if there's no more room, allocate more */
    if (table->occupancy == table->capacity) {
        table->capacity *= 2;
        table->chain = (DiagHashCell*)realloc(table->chain, table->capacity * sizeof(DiagHashCell));
        if (table->chain == NULL)
            return 0;
    }

    cell = table->chain + table->occupancy;
    cell->diag = diag;
    cell->level = level;
    cell->hit_len = len;
    cell->hit_saved = hit_saved;
    cell->next = table->backbone[bucket];
    table->backbone[bucket] = table->occupancy;
    table->occupancy++;

    return 1;
}

///** Test to see if seed->q_off exists in lookup table
// * @param lookup_wrap The lookup table wrap structure [in]
// * @param subject Subject sequence data [in]
// * @s_off The starting offset of the seed [in]
// * @lut_word_length The length of the lookup word [in]
// */
//static NCBI_INLINE Boolean
//s_IsSeedMasked(const LookupTableWrap   * lookup_wrap,
//               const BLAST_SequenceBlk * subject,
//               Int4                      s_off,
//               Int4                      lut_word_length,
//               Int4                      q_pos)
//{
//    Uint1 *s  = subject->sequence + s_off / COMPRESSION_RATIO;
//    Int4 shift = 2* (16 - s_off % COMPRESSION_RATIO - lut_word_length);
//    Int4 index = (s[0] << 24 | s[1] << 16 | s[2] << 8 | s[3]) >> shift;
//    return !(((T_Lookup_Callback)(lookup_wrap->lookup_callback))
//                                         (lookup_wrap, index, q_pos));
//}
//
/** Check the mini-extended word against masked query regions, and do right
 * extension if necessary.
 * @param query Query sequence data [in]
 * @param subject Subject sequence data [in]
 * @param q_off Query offset [in][out]
 * @param s_off Subject offset [in][out]
 * @param locations of the masked query regions [in]
 * @param query_info of the masked query regions [in]
 * @param s_range the open bound of subject region [in]
 * @param word_length length of word [in]
 * @param lut_word_length length of lookup table word [in]
 * @param check_double check to see if it is a double word [in]
 * @param extended if successful, the actual bases extended [out]
 * @return 0,1,2  for non-word, single word, and double word
 */
static Int4
s_TypeOfWord(BLAST_SequenceBlk * query,
             BLAST_SequenceBlk * subject,
             Int4 *q_off, Int4 *s_off,
             BlastSeqLoc* locations, 
             BlastQueryInfo * query_info,
             Uint4 s_range,
             Uint4 word_length,
             Uint4 lut_word_length,
             const LookupTableWrap* lookup_wrap,
             Boolean check_double,
             Int4 * extended)
{
    Int4 context, q_range;
    Int4 ext_to, ext_max;
    Int4 q_end = *q_off + word_length;
    Int4 s_end = *s_off + word_length;
    Int4 s_pos, q_pos;

    *extended = 0;

    /* No need to check if mini-extension is not performed.
       It turns out that we may skip checking for double-hit in 2-hit 
       algo case as well -- they will come in as 2 hits naturally*/
    if (word_length == lut_word_length) return 1;

    /* Find the query context boundary */
    context = BSearchContextInfo(q_end, query_info);
    q_range = query_info->contexts[context].query_offset 
            + query_info->contexts[context].query_length;

    /* check query mask at two ends */
    if (locations) {
        /* check for right end first */
        if (s_IsSeedMasked(lookup_wrap, subject, 
                           s_end - lut_word_length, 
                           lut_word_length, 
                           q_end - lut_word_length)) return 0;

        /* search for valid left end and reposition q_off */
        for (; TRUE; ++(*s_off), ++(*q_off)) {
            if (!s_IsSeedMasked(lookup_wrap, subject,
                                *s_off, lut_word_length, *q_off)) break;
        }
    }

    ext_to = word_length - (q_end - (*q_off));
    ext_max = MIN(q_range - q_end, s_range - s_end);

    /* shift to the right, and check query mask inside */
    if (ext_to || locations) {

        if (ext_to > ext_max) return 0;
        q_end += ext_to;
        s_end += ext_to;

        for (s_pos = s_end - lut_word_length, 
             q_pos = q_end - lut_word_length;
             s_pos > *s_off; 
             s_pos -= lut_word_length,
             q_pos -= lut_word_length) {
             if (s_IsSeedMasked(lookup_wrap, subject,
                           s_pos, lut_word_length, q_pos)) return 0;
        }

        (*extended) = ext_to;
    }

    /* if we get here, single word check is passed */
    if (!check_double) return 1;

    /* do right extension to double word
       Note: unlike the single word check, here we need to determine the 
       precise value of maximal possible *extend */
    ext_to += word_length;
    ext_max = MIN(ext_max, ext_to);

    /* try seed by seed */
    for (s_pos = s_end, q_pos = q_end; 
         *extended + lut_word_length <= ext_max; 
         s_pos += lut_word_length, 
         q_pos += lut_word_length,
         (*extended) += lut_word_length) {
         if (s_IsSeedMasked(lookup_wrap, subject, s_pos, 
                            lut_word_length, q_pos)) break;
    }

    /* try base by base */
    s_pos -= (lut_word_length - 1);
    q_pos -= (lut_word_length - 1);
    while (*extended < ext_max) {
        if (s_IsSeedMasked(lookup_wrap, subject, s_pos,
                               lut_word_length, q_pos)) return 1;
        (*extended)++;
        ++s_pos;
        ++q_pos;
    }
        
    return ((ext_max == ext_to) ? 2 : 1);
}

/** Perform ungapped extension given an offset pair, and save the initial 
 * hit information if the hit qualifies. This function assumes that the
 * exact match has already been extended to the word size parameter. It also
 * supports a two-hit version, where extension is performed only after two hits
 * are detected within a window.
 * @param query The query sequence [in]
 * @param subject The subject sequence [in]
 * @param q_off The offset in the query sequence [in]
 * @param s_off The offset in the subject sequence [in]
 * @param query_mask Structure containing query mask ranges [in]
 * @param query_info Structure containing query context ranges [in]
 * @param s_range Subject range [in]
 * @param word_length The length of the hit [in]
 * @param word_lut_length The length of the lookup table word [in]
 * @param word_params The parameters related to initial word extension [in]
 * @param matrix the substitution matrix for ungapped extension [in]
 * @param hash_table Structure containing initial hits [in] [out]
 * @param init_hitlist The structure containing information about all 
 *                     initial hits [in] [out]
 * @return 1 if hit was extended, 0 if not
 */
static Int4
s_BlastnDiagHashExtendInitialHit(BLAST_SequenceBlk * query,
                               BLAST_SequenceBlk * subject, 
                               Int4 q_off, Int4 s_off, 
                               BlastSeqLoc* query_mask,
                               BlastQueryInfo * query_info,
                               Int4 s_range,
                               Int4 word_length, Int4 lut_word_length,
                               const LookupTableWrap * lut,
                               const BlastInitialWordParameters * word_params,
                               Int4 ** matrix,
                               BLAST_DiagHash * hash_table,
                               BlastInitHitList * init_hitlist)
{
    Int4 diag;
    Int4 s_end, s_off_pos, s_end_pos, s_l;
    Int4 word_type = 0;
    Int4 extended = 0;
    BlastUngappedData *ungapped_data;
    BlastUngappedData dummy_ungapped_data;
    Int4 window_size = word_params->options->window_size;
    Int4 hit_ready = 1; 
    Int4 last_hit, hit_saved = 0;
    BlastUngappedCutoffs *cutoffs = NULL;
    Boolean two_hits = (window_size > 0);
    Boolean off_found = FALSE;
    Int4 Delta = MIN(word_params->options->scan_range, window_size - word_length);
    Int4 rc;

    diag = s_off - q_off;
    s_end = s_off + word_length;
    s_off_pos = s_off + hash_table->offset;
    s_end_pos = s_end + hash_table->offset;

    rc = s_BlastDiagHashRetrieve(hash_table, diag, &last_hit, &s_l, &hit_saved);

    /* if there is no record in hashtable, we set last_hit to be a very negative number */
    if(!rc)  last_hit = 0;

    /* hit within the explored area should be rejected*/
    if (s_off_pos < last_hit) return 0;

    if (two_hits && (hit_saved || s_end_pos > last_hit + window_size )) {
        /* check the masks for the word; also check to see if this
           first hit qualifies for a double-hit */
        word_type = s_TypeOfWord(query, subject, &q_off, &s_off,
                                 query_mask, query_info, s_range,
                                 word_length, lut_word_length, lut, TRUE, &extended);
        if (!word_type) return 0;
        /* update the right end*/
        s_end += extended;
        s_end_pos += extended;

        /* for single word, also try off diagonals */
        if (word_type == 1) {
            /* try off-diagonal */
            Int4 s_a = s_off_pos + word_length - window_size;
            Int4 s_b = s_end_pos - 2 * word_length;
            Int4 delta;
            if (Delta < 0) Delta = 0;
            for (delta = 1; delta <= Delta; ++delta) {
                Int4 off_s_end = 0;
                Int4 off_s_l = 0;
                Int4 off_hit_saved = 0;
                Int4 off_rc = s_BlastDiagHashRetrieve(hash_table, diag + delta, 
                              &off_s_end, &off_s_l, &off_hit_saved);
                if ( off_rc
                  && off_s_l
                  && off_s_end - delta >= s_a
                  && off_s_end - off_s_l <= s_b) {
                     off_found = TRUE;
                     break;
                }
                off_rc = s_BlastDiagHashRetrieve(hash_table, diag - delta, 
                              &off_s_end, &off_s_l, &off_hit_saved);
                if ( off_rc
                  && off_s_l
                  && off_s_end >= s_a
                  && off_s_end - off_s_l + delta <= s_b) {
                     off_found = TRUE;
                     break;
                }
            }
            if (!off_found) {
                /* This is a new hit */
                hit_ready = 0;
            }
        }
    } else {
        /* check the masks for the word */
        if (!s_TypeOfWord(query, subject, &q_off, &s_off,
                          query_mask, query_info, s_range,
                          word_length, lut_word_length, lut, FALSE, &extended)) return 0;
        /* update the right end*/
        s_end += extended;
        s_end_pos += extended;
    }

    if (hit_ready) {
        if (word_params->ungapped_extension) {
            /* Perform ungapped extension */
            Int4 context = BSearchContextInfo(q_off, query_info);
            cutoffs = word_params->cutoffs + context;
            ungapped_data = &dummy_ungapped_data;

            /* 
             * Skip use of the scoring table and go straight to the matrix
             * based extension if matrix_only_scoring is set.  Used by
             * app rmblastn.
             * -RMH-
             */
            if ( word_params->options->program_number == eBlastTypeBlastn &&                          word_params->matrix_only_scoring )
            {
                s_NuclUngappedExtendExact(query, subject, matrix, q_off,
                                  s_off, -(cutoffs->x_dropoff), ungapped_data);
            }else {
                s_NuclUngappedExtend(query, subject, matrix, q_off, s_end,
                                 s_off, -(cutoffs->x_dropoff),
                                 ungapped_data,
                                 word_params->nucl_score_table,
                                 cutoffs->reduced_nucl_cutoff_score);
            }

            if (off_found || ungapped_data->score >= cutoffs->cutoff_score) {
                BlastUngappedData *final_data =
                    (BlastUngappedData *) malloc(sizeof(BlastUngappedData));
                *final_data = *ungapped_data;
                BLAST_SaveInitialHit(init_hitlist, q_off, s_off, final_data);
                s_end_pos = ungapped_data->length + ungapped_data->s_start 
                          + hash_table->offset;
            } else {
                hit_ready = 0;
            }
        } else {
            ungapped_data = NULL;
            BLAST_SaveInitialHit(init_hitlist, q_off, s_off, ungapped_data);
        }
    } 
    
    s_BlastDiagHashInsert(hash_table, diag, s_end_pos, 
                          (hit_ready) ? 0 : s_end_pos - s_off_pos,
                          hit_ready, s_off_pos, window_size + Delta + 1);

    return hit_ready;
}


/** Perform ungapped extension given an offset pair, and save the initial 
 * hit information if the hit qualifies. This function assumes that the
 * exact match has already been extended to the word size parameter. It also
 * supports a two-hit version, where extension is performed only after two hits
 * are detected within a window.
 * @param query The query sequence [in]
 * @param subject The subject sequence [in]
 * @param q_off The offset in the query sequence [in]
 * @param s_off The offset in the subject sequence [in]
 * @param query_mask Structure containing query mask ranges [in]
 * @param query_info Structure containing query context ranges [in]
 * @param s_range Subject range [in]
 * @param word_length The length of the hit [in]
 * @param word_lut_length The length of the lookup table word [in]
 * @param word_params The parameters related to initial word extension [in]
 * @param matrix the substitution matrix for ungapped extension [in]
 * @param diag_table Structure containing diagonal table with word extension 
 *                   information [in]
 * @param init_hitlist The structure containing information about all 
 *                     initial hits [in] [out]
 * @return 1 if hit was extended, 0 if not
 */
static Int4
s_BlastnDiagTableExtendInitialHit(BLAST_SequenceBlk * query,
                             BLAST_SequenceBlk * subject, 
                             Int4 q_off, Int4 s_off, 
                             BlastSeqLoc* query_mask,
                             BlastQueryInfo * query_info,
                             Int4 s_range,
                             Int4 word_length, Int4 lut_word_length,
                             const LookupTableWrap * lut,
                             const BlastInitialWordParameters * word_params,
                             Int4 ** matrix, 
                             BLAST_DiagTable * diag_table,
                             BlastInitHitList * init_hitlist)
{
    Int4 diag, real_diag;
    Int4 s_end, s_off_pos, s_end_pos;
    Int4 word_type = 0;
    Int4 extended = 0;
    BlastUngappedData *ungapped_data;
    BlastUngappedData dummy_ungapped_data;
    Int4 window_size = word_params->options->window_size;
    Int4 hit_ready = 1;
    Int4 last_hit, hit_saved;
    DiagStruct *hit_level_array;
    BlastUngappedCutoffs *cutoffs = NULL;
    Boolean two_hits = (window_size > 0);
    Boolean off_found = FALSE;
    Int4 Delta = MIN(word_params->options->scan_range, window_size - word_length);

    hit_level_array = diag_table->hit_level_array;
    ASSERT(hit_level_array);

    diag = s_off + diag_table->diag_array_length - q_off;
    real_diag = diag & diag_table->diag_mask;
    last_hit = hit_level_array[real_diag].last_hit;
    hit_saved = hit_level_array[real_diag].flag;
    s_end = s_off + word_length;
    s_off_pos = s_off + diag_table->offset;
    s_end_pos = s_end + diag_table->offset;

    /* hit within the explored area should be rejected*/
    if (s_off_pos < last_hit) return 0;

    if (two_hits && (hit_saved || s_end_pos > last_hit + window_size )) {
        /* check the masks for the word; also check to see if this
           first hit qualifies for a double-word */
        word_type = s_TypeOfWord(query, subject, &q_off, &s_off,
                                 query_mask, query_info, s_range, 
                                 word_length, lut_word_length, lut, TRUE, &extended);
        if (!word_type) return 0;
        /* update the right end*/
        s_end += extended;
        s_end_pos += extended;

        /* for single word, also try off diagonals */
        if (word_type == 1) {
            /* try off-diagonals */
            Int4 orig_diag = real_diag + diag_table->diag_array_length;
            Int4 s_a = s_off_pos + word_length - window_size;
            Int4 s_b = s_end_pos - 2 * word_length;
            Int4 delta;
            if (Delta < 0) Delta = 0;
            for (delta = 1; delta <= Delta ; ++delta) {
                Int4 off_diag  = (orig_diag + delta) & diag_table->diag_mask;
                Int4 off_s_end = hit_level_array[off_diag].last_hit;
                Int4 off_s_l   = diag_table->hit_len_array[off_diag];
                if ( off_s_l
                 && off_s_end - delta >= s_a 
                 && off_s_end - off_s_l <= s_b) {
                    off_found = TRUE;
                    break;
                }
                off_diag  = (orig_diag - delta) & diag_table->diag_mask;
                off_s_end = hit_level_array[off_diag].last_hit;
                off_s_l   = diag_table->hit_len_array[off_diag];
                if ( off_s_l
                 && off_s_end >= s_a 
                 && off_s_end - off_s_l + delta <= s_b) {
                    off_found = TRUE;
                    break;
                }
            }
            if (!off_found) {
                /* This is a new hit */
                hit_ready = 0;
            }
        }
    } else {
        /* check the masks for the word */
        if(!s_TypeOfWord(query, subject, &q_off, &s_off,
                        query_mask, query_info, s_range, 
                        word_length, lut_word_length, lut, FALSE, &extended)) return 0;
        /* update the right end*/
        s_end += extended;
        s_end_pos += extended;
    }
       
    if (hit_ready) {
        if (word_params->ungapped_extension) {
            Int4 context = BSearchContextInfo(q_off, query_info);
            cutoffs = word_params->cutoffs + context;
            ungapped_data = &dummy_ungapped_data;

            /* 
             * Skip use of the scoring table and go straight to the matrix
             * based extension if matrix_only_scoring is set.  Used by
             * app rmblastn.
             * -RMH-
             */
            if ( word_params->options->program_number == eBlastTypeBlastn &&
                 word_params->matrix_only_scoring )
            {
               s_NuclUngappedExtendExact(query, subject, matrix, q_off,
                                  s_off, -(cutoffs->x_dropoff), ungapped_data);
            }else {
               s_NuclUngappedExtend(query, subject, matrix, q_off, s_end, s_off,
                                 -(cutoffs->x_dropoff), ungapped_data,
                                 word_params->nucl_score_table,
                                 cutoffs->reduced_nucl_cutoff_score);
            }

            if (off_found || ungapped_data->score >= cutoffs->cutoff_score) {
                BlastUngappedData *final_data =
                    (BlastUngappedData *) malloc(sizeof(BlastUngappedData));
                *final_data = *ungapped_data;
                BLAST_SaveInitialHit(init_hitlist, q_off, s_off, final_data);
                s_end_pos = ungapped_data->length + ungapped_data->s_start
                          + diag_table->offset;
            } else {
                hit_ready = 0;
            }
        } else {
            ungapped_data = NULL;
            BLAST_SaveInitialHit(init_hitlist, q_off, s_off, ungapped_data);
        }
    } 

    hit_level_array[real_diag].last_hit = s_end_pos;
    hit_level_array[real_diag].flag = hit_ready;
    if (two_hits) {
        diag_table->hit_len_array[real_diag] = (hit_ready) ? 0 : s_end_pos - s_off_pos;
    }

    return hit_ready;
}

//#endif  // __UNGPAPPED_EXTENSION_FUNCTIONS_H__