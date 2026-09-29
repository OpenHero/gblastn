#ifndef GBLASTN_MINI_EXTENSION_CUH
#define GBLASTN_MINI_EXTENSION_CUH
// Same exact-match extension as NCBI 2.2.28+/the original G-BLASTN kernel.
// Query sentinels and explicit limits keep reverse-strand/context boundaries
// from crossing and prevent even speculative reads outside the allocation.
__global__ void kernel_s_BlastNaExtend_withoutHash_v3(const Uint1* subject,
    const Uint1* query, Int4 lut_word_length, Int4 ext_to, Int4 num_hits,
    const BlastOffsetPair* input, BlastOffsetPair* output, Int4 s_range, Int4 q_range)
{
    Int4 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_hits) return;
    output[i].qs_offsets.q_off = Uint4(-1);
    output[i].qs_offsets.s_off = 0;
    Int4 s = input[i].qs_offsets.s_off, q = input[i].qs_offsets.q_off;
    Int4 left = 0;
    while (left < ext_to && left < s && left < q) {
        Int4 p = s - left - 1;
        if (((subject[p / 4] >> (6 - 2 * (p % 4))) & 3) != query[q - left - 1]) break;
        ++left;
    }
    Int4 remaining = ext_to - left;
    if (s + lut_word_length > s_range - remaining ||
        q + lut_word_length > q_range - remaining) return;
    for (Int4 r = 0; r < remaining; ++r) {
        Int4 p = s + lut_word_length + r;
        if (((subject[p / 4] >> (6 - 2 * (p % 4))) & 3) != query[q + lut_word_length + r]) return;
    }
    output[i].qs_offsets.q_off = q - left;
    output[i].qs_offsets.s_off = s - left;
}
#endif
