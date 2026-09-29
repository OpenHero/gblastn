#ifndef GBLASTN_MB_SCAN_CUH
#define GBLASTN_MB_SCAN_CUH
__device__ inline Uint4 gpu_mb_chain(Int4 q, const Int4* next,
    Uint4 next_length, Uint4 chain_limit, BlastOffsetPair* out,
    Int4 subject_offset, Uint4* error)
{
    Uint4 count = 0;
    while (q != 0) {
        if (q < 0 || Uint4(q) >= next_length || count >= chain_limit) {
            atomicExch(error, 1u); return 0;
        }
        if (out) { out[count].qs_offsets.q_off = q - 1; out[count].qs_offsets.s_off = subject_offset; }
        ++count;
        q = next[q];
    }
    return count;
}
#endif
