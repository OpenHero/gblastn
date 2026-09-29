#ifndef GBLASTN_SMALL_SCAN_CUH
#define GBLASTN_SMALL_SCAN_CUH
// Read exactly the requested bases. No unaligned Uint4 loads or tail overreads.
__device__ inline Uint4 gpu_word(const Uint1* subject, Int4 offset, Int4 length)
{
    Uint4 word = 0;
    for (Int4 j = 0; j < length; ++j) {
        const Int4 pos = offset + j;
        word = (word << 2) | ((subject[pos / 4] >> (6 - 2 * (pos % 4))) & 3);
    }
    return word;
}

// One thread owns a subject position. Int2 stays SIGNED: -1 is empty and
// values below -1 address the terminated overflow chain.
__device__ inline Uint4 gpu_small_chain(Int2 entry, const Int2* overflow,
    Uint4 overflow_length, Uint4 chain_limit, BlastOffsetPair* out,
    Int4 subject_offset, Uint4* error)
{
    if (entry == -1) return 0;
    if (entry >= 0) {
        if (out) { out[0].qs_offsets.q_off = entry; out[0].qs_offsets.s_off = subject_offset; }
        return 1;
    }
    Uint4 at = -Int4(entry), count = 0;
    for (;;) {
        if (at >= overflow_length) { atomicExch(error, 1u); return 0; }
        Int2 q = overflow[at++];
        if (q < 0) return count;
        if (count >= chain_limit) { atomicExch(error, 1u); return 0; }
        if (out) { out[count].qs_offsets.q_off = q; out[count].qs_offsets.s_off = subject_offset; }
        ++count;
    }
}
#endif
