#ifndef __GPU_BLAST_LOOKUP_HASH_V3_H__
#define __GPU_BLAST_LOOKUP_HASH_V3_H__

__global__ 
	void kernel_lookupInBigHashTable_v3( 
	Int4 * hashtable,
	Int4 * next_pos,
	Uint4 num_hits,
	BlastOffsetPair * offset_pairs, BlastOffsetPair * exact_offset_pairs,
	unsigned int *total_hits_num,
	Uint4 next_pos_len)
{						
	Uint4 index = blockIdx.x*blockDim.x +threadIdx.x;
	if (index< num_hits)
	{
		BlastOffsetPair qs_offset_pair = offset_pairs[index];
		Uint4 s_index = qs_offset_pair.qs_offsets.q_off;
		Uint4 s_off = qs_offset_pair.qs_offsets.s_off;
		Uint4 q_off = hashtable[s_index];
		while(q_off)
		{	 
			Uint4 i = atomicAdd(total_hits_num, 1);
			exact_offset_pairs[i].qs_offsets.q_off = q_off - 1;
			exact_offset_pairs[i].qs_offsets.s_off = s_off;
			if (q_off < next_pos_len)
			{
				q_off = next_pos[q_off];
			}else
			{
				break;
			}
		}
	}
}

#endif
