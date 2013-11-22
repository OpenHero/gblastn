#ifndef __GPU_BLASTN_SMALL_LOOKUPTABLE_KERNEL_V3_H__
#define __GPU_BLASTN_SMALL_LOOKUPTABLE_KERNEL_V3_H__

__global__ 
	void kernel_lookupInSmallTable_v3( 
	Int2 * overflowtable,
	Uint4 num_hits,
	BlastOffsetPair * over_offset_pairs, BlastOffsetPair * exact_offset_pairs,
	unsigned int *total_hits_num,
	Uint4 overflowtable_len)
{						
	Uint4 index = blockIdx.x*blockDim.x +threadIdx.x;
	if (index< num_hits)
	{
		BlastOffsetPair qs_offset_pair = over_offset_pairs[index];
		Uint4 src_off = qs_offset_pair.qs_offsets.q_off;
		Uint4 s_off = qs_offset_pair.qs_offsets.s_off;
		Int4 s_index = overflowtable[src_off++];

		do {
			Uint4 i = atomicAdd(total_hits_num, 1);
			exact_offset_pairs[i].qs_offsets.q_off = s_index;
			exact_offset_pairs[i].qs_offsets.s_off = s_off;
			if (src_off <= overflowtable_len)
			{
				s_index = overflowtable[src_off++];
			}
			else
			{
				break;
			}
		} while (s_index >= 0);
	}
}
#endif	//__GPU_BLASTN_SMALL_LOOKUPTABLE_KERNEL_H__