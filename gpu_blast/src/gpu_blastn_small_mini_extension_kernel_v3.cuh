#ifndef __GPU_BLASTN_SMALL_MINI_EXTENSION_KERNEL_V3_H__
#define __GPU_BLASTN_SMALL_MINI_EXTENSION_KERNEL_V3_H__
//////////////////////////////////////////////////////////////////////////
/** Entry i of this list gives the number of pairs of
 * bits that are zero in the bit pattern of i, looking 
 * from right to left
 */
__constant__ Uint1 s_gpu_ExactMatchExtendLeft[256] = {
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
__constant__ Uint1 s_gpu_ExactMatchExtendRight[256] = {
4, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
};

__device__ __inline__ Int4 gpu_BSearchContextInfo(Int4 n, BlastContextInfo * A_contexts, Int4 A_last_context)
{
	Int4 m=0, b=0, e=0, size=0;

	size = A_last_context+1;

	b = 0;
	e = size;
	while (b < e - 1) {
		m = (b + e) / 2;
		if (A_contexts[m].query_offset > n)
			e = m;
		else
			b = m;
	}
	return b;
}


/** CUDA Kernel Perform exact match extensions on the hits retrieved from
* blastn/megablast lookup tables, assuming an arbitrary number of bases 
* in a lookup and arbitrary start offset of each hit.
* @param offset_pairs Array of query and subject offsets [in]
* @param num_hits Size of the above arrays [in]
* @param offset_pairs Array of query and subject offsets after exact match extensions [out]
* @param query Query sequence data [in]
* @param subject Subject sequence data [in]
* @param init_hitlist Structure to keep the extended hits. 
*                     Must be allocated outside of this function [in] [out]
* @param s_range The subject range [in]
* @param exact_seeds_num the number of hits after exact match extensions [out]
*/
__global__	void kernel_s_BlastSmallExtend_v3( Uint1 * subject_sequence,
	Uint1 * query_compressed_nuc_seq_start,
	Int4 word_length,
	Int4 lut_word_length,
	Int4 num_hits,
	BlastOffsetPair * offset_pairs, BlastOffsetPair * exact_offset_pairs,
	unsigned int *exact_hits_num,
	Uint4 s_range,
	BlastContextInfo * A_contexts,
	Int4 A_last_context)
{
	Int4 index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < num_hits) 
	{
		Uint1 *q = query_compressed_nuc_seq_start + 3;
		Uint1 *s = subject_sequence;

		Int4 s_offset = offset_pairs[index].qs_offsets.s_off;
        Int4 q_offset = offset_pairs[index].qs_offsets.q_off;
        Int4 s_off;
        Int4 q_off;
        Int4 ext_left = 0;
        Int4 ext_right = 0;
        Int4 context = gpu_BSearchContextInfo(q_offset, A_contexts, A_last_context);
        Int4 q_start = A_contexts[context].query_offset;
        Int4 q_range = q_start + A_contexts[context].query_length;

        Int4 ext_max = MIN(MIN(word_length - lut_word_length, s_offset), q_offset - q_start);

//#if 0
//		if ((s_offset == 107533824)&& (q_offset == 641))
//		{
//			exact_offset_pairs[0].qs_offsets.s_off = s_offset;   
//			exact_offset_pairs[0].qs_offsets.q_off = q_offset;
//			exact_offset_pairs[1].qs_offsets.s_off = context;   
//			exact_offset_pairs[1].qs_offsets.q_off = q_start;
//			exact_offset_pairs[2].qs_offsets.s_off = q_range;   
//			exact_offset_pairs[2].qs_offsets.q_off = ext_max;
//		}
//#endif

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
            Uint1 bases = s_gpu_ExactMatchExtendLeft[q_byte ^ s_byte];
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
            Uint1 bases = s_gpu_ExactMatchExtendRight[q_byte ^ s_byte];
            ext_right += bases;
            if (bases < 4)
                break;
            q_off += 4;
            s_off += 4;
        }
        ext_right = MIN(ext_right, ext_max);

        if ((ext_left + ext_right) >= word_length)
		{
			q_offset -= ext_left;
			s_offset -= ext_left;

			unsigned int hit_index = atomicAdd(exact_hits_num, 1);      //找到一个hits，计数器加1
			exact_offset_pairs[hit_index].qs_offsets.s_off = s_offset;   
			exact_offset_pairs[hit_index].qs_offsets.q_off = q_offset;
		}
	}

}
#endif