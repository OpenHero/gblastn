#ifndef GBLASTN_LOOKUP_CUH
#define GBLASTN_LOOKUP_CUH
// Count and emit are separate launches, with a prefix sum between them. This
// preserves the CPU scanner's subject order AND the table's query-chain order.
// No inter-thread shared memory, warp lockstep assumption, or append overflow.
template<bool Small, bool Emit>
__global__ void gpu_scan_lookup(const Uint1* subject, Int4 start, Int4 step,
    Int4 word_length, Uint4 positions, const Int2* backbone,
    const Int2* overflow, Uint4 overflow_length, const Int4* hash,
    const Int4* next, Uint4 next_length, Uint4 chain_limit,
    Uint4* counts, const Uint4* prefix, BlastOffsetPair* output, Uint4* error)
{
    Uint4 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= positions) return;
    Int4 s = start + Int4(i) * step;
    Uint4 word = gpu_word(subject, s, word_length);
    BlastOffsetPair* out = Emit ? output + prefix[i] : NULL;
    Uint4 count = Small
        ? gpu_small_chain(backbone[word], overflow, overflow_length, chain_limit, out, s, error)
        : gpu_mb_chain(hash[word], next, next_length, chain_limit, out, s, error);
    if (!Emit) counts[i] = count;
}
#endif
