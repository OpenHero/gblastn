#ifndef __GPU_BLASTN_MINI_EXTENSION_KERNEL_V3_CUH__
#define __GPU_BLASTN_MINI_EXTENSION_KERNEL_V3_CUH__

__global__	void kernel_s_BlastNaExtend_withoutHash_v3( 
	Uint1 * subject,
	Uint1 * query,
	Int4 lut_word_length,
	Int4 ext_to,
	Int4 num_hits,
	BlastOffsetPair * offset_pairs, BlastOffsetPair * exact_offset_pairs,
	unsigned int *exact_hits_num,
	Uint4 s_range,
	Uint4 q_range)
{
	//bool is_update = false;
	Int4 index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < num_hits) 
	{
		Uint4 s_offset = offset_pairs[index].qs_offsets.s_off;
		Uint4 q_offset = offset_pairs[index].qs_offsets.q_off;
		/* begin with the left extension; the initialization is slightly
		faster. Point to the first base of the lookup table hit and
		work backwards */
		if (s_offset <= s_range)
		{
			Int4 ext_left = 0;
			Uint4 s_off = s_offset;
			Uint1 *q = query + q_offset;
			//Uint1 *q_lat = query + q_range;
			Uint1 *s = subject + s_off / COMPRESSION_RATIO;

			if (q_offset < (q_range)) // add the range limited.
			{
				for (; ext_left < MIN(ext_to, s_offset); ++ext_left) {
					s_off--;
					q--;
					if (q < query) break; // added by kyzhao for range limited
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
					if ((s_off + ext_to - ext_left) > s_range) return;
					
						q = query + q_offset + lut_word_length;
						s = subject + s_off / COMPRESSION_RATIO;

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
						return;
				}

				q_offset -= ext_left;
				s_offset -= ext_left;
				unsigned int hit_index = atomicAdd(exact_hits_num, 1);      //找到一个hits，计数器加1
				exact_offset_pairs[hit_index].qs_offsets.s_off = s_offset;   
				exact_offset_pairs[hit_index].qs_offsets.q_off = q_offset;

			} 
		}
	}
}

#endif