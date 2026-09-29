// Portable scalar emulation of the CUDA kernel bodies, with independent
// sequence-based expectations. This checks arithmetic/bounds only; it is NOT
// evidence of CUDA compilation, scheduling, or device execution.
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>
typedef uint8_t Uint1;
typedef int16_t Int2;
typedef int32_t Int4;
typedef uint32_t Uint4;
struct BlastOffsetPair { struct { Uint4 q_off, s_off; } qs_offsets; };
struct GpuQueryContext { Int4 query_offset, query_length; };
struct Dim { int x; } threadIdx = {0}, blockIdx = {0}, blockDim = {1};
#define __global__
#define __device__
#define __constant__
#define COMPRESSION_RATIO 4
#define MIN(a,b) std::min((a),(b))
Uint4 atomicExch(Uint4* p, Uint4 v) { Uint4 old = *p; *p = v; return old; }
#include "../c++/src/algo/blast/gpu_blast/gpu_blastn_small_scan_kernel_v3.cuh"
#include "../c++/src/algo/blast/gpu_blast/gpu_blastn_mb_scan_kernel_v3.cuh"
#include "../c++/src/algo/blast/gpu_blast/gpu_blastn_lookup_hash_kernel_v3.cuh"
#include "../c++/src/algo/blast/gpu_blast/gpu_blastn_small_mini_extension_kernel_v3.cuh"
#include "../c++/src/algo/blast/gpu_blast/gpu_blastn_mini_extension_kernel_v3.cuh"

std::vector<Uint1> pack(const std::vector<Uint1>& s)
{
    std::vector<Uint1> result((s.size()+3)/4);
    for (size_t i=0; i<s.size(); ++i) result[i/4] |= s[i] << (6-2*(i%4));
    return result;
}
std::vector<Uint1> sliding(const std::vector<Uint1>& s)
{
    std::vector<Uint1> result(s.size()+3);
    for (int i=-3; i<int(s.size()); ++i)
        for (int j=0; j<4; ++j)
            if (i+j>=0 && i+j<int(s.size())) result[i+3] |= s[i+j] << (6-2*j);
    return result;
}
BlastOffsetPair reference(const std::vector<Uint1>& s, const std::vector<Uint1>& q,
                          int so, int qo, int width, int word)
{
    int left=0;
    while (left<word-width && left<so && left<qo && s[so-left-1]==q[qo-left-1]) ++left;
    int need=word-width-left;
    if (so+width+need>int(s.size()) || qo+width+need>int(q.size())) return {{Uint4(-1),0}};
    for (int r=0;r<need;++r) if (s[so+width+r]!=q[qo+width+r]) return {{Uint4(-1),0}};
    return {{Uint4(qo-left),Uint4(so-left)}};
}
int main()
{
    // Signed minimum, negative sentinel, invalid chains, order and exact capacity.
    std::vector<Int2> overflow(32772, -1);
    overflow[32768]=12; overflow[32769]=3; overflow[32770]=27;
    BlastOffsetPair out[4] = {}; Uint4 error=0;
    assert(gpu_small_chain(-32768, overflow.data(), overflow.size(), 3, out, 44, &error)==3);
    assert(!error && out[0].qs_offsets.q_off==12 && out[1].qs_offsets.q_off==3 && out[2].qs_offsets.q_off==27);
    gpu_small_chain(-32768, overflow.data(), overflow.size(), 2, out, 44, &error);
    assert(error); error=0;
    assert(gpu_small_chain(-1, NULL, 0, 1, out, 0, &error)==0 && !error);
    assert(gpu_small_chain(32767, NULL, 0, 1, out, 0, &error)==1 && out[0].qs_offsets.q_off==32767);
    Int4 next[]={0,0,4,0,1};
    assert(gpu_mb_chain(2,next,5,3,out,9,&error)==3 && !error);
    assert(out[0].qs_offsets.q_off==1 && out[1].qs_offsets.q_off==3 && out[2].qs_offsets.q_off==0);
    next[1]=2; gpu_mb_chain(2,next,5,3,out,9,&error); assert(error);

    std::mt19937 rng(2828);
    // End-to-end count/prefix/emit ordering for arbitrary starts and strides.
    for (int trial=0; trial<100; ++trial) {
        const int width=4;
        std::vector<Uint1> q(150), subject(103);
        for (auto& b:q) b=rng()%4;
        for (auto& b:subject) b=rng()%4;
        std::vector<std::vector<Int4>> chains(256);
        for (int qo=0;qo<=int(q.size())-width;++qo) {
            Uint4 word=0;
            for (int j=0;j<width;++j) word=word*4+q[qo+j];
            chains[word].push_back(qo);
        }
        std::vector<Int2> backbone(256,-1), over(2,-1);
        Uint4 limit=1;
        for (size_t w=0;w<chains.size();++w) {
            limit=std::max(limit,Uint4(chains[w].size()));
            if (chains[w].size()==1) backbone[w]=chains[w][0];
            else if (!chains[w].empty()) {
                backbone[w]=-Int2(over.size());
                for (auto qo:chains[w]) over.push_back(qo);
                over.push_back(-1);
            }
        }
        auto packed=pack(subject);
        int start=trial%5, step=1+trial%7;
        Uint4 positions=(int(subject.size())-width-start)/step+1;
        std::vector<Uint4> count(positions), prefix(positions);
        std::vector<BlastOffsetPair> emitted(positions*limit), expected;
        error=0;
        for (Uint4 i=0;i<positions;++i) {
            threadIdx.x=i;
            gpu_scan_lookup<true,false>(packed.data(),start,step,width,positions,
                backbone.data(),over.data(),over.size(),NULL,NULL,0,limit,
                count.data(),prefix.data(),emitted.data(),&error);
            int so=start+i*step;
            Uint4 word=0;
            for(int j=0;j<width;++j) word=word*4+subject[so+j];
            for(auto qo:chains[word]) expected.push_back({{Uint4(qo),Uint4(so)}});
        }
        Uint4 total=0;
        for (Uint4 i=0;i<positions;++i) { prefix[i]=total; total+=count[i]; }
        assert(total==expected.size());
        for (Uint4 i=0;i<positions;++i) {
            threadIdx.x=i;
            gpu_scan_lookup<true,true>(packed.data(),start,step,width,positions,
                backbone.data(),over.data(),over.size(),NULL,NULL,0,limit,
                count.data(),prefix.data(),emitted.data(),&error);
        }
        assert(!error);
        for (Uint4 i=0;i<total;++i) {
            assert(emitted[i].qs_offsets.q_off==expected[i].qs_offsets.q_off);
            assert(emitted[i].qs_offsets.s_off==expected[i].qs_offsets.s_off);
        }
    }
    threadIdx.x=0;
    unsigned checks=0;
    for (int trial=0; trial<20000; ++trial) {
        int ns=9+rng()%130, nq=9+rng()%90;
        std::vector<Uint1> s(ns),q(nq);
        for (auto& b:s) b=rng()%4;
        for (auto& b:q) b=rng()%4;
        int width=8, word=width+1+rng()%8;
        int so=rng()%(ns-width+1), qo=rng()%(nq-width+1);
        for (int j=0;j<width;++j) q[qo+j]=s[so+j];
        if (trial%2==0) { // Include exact extensions and tails, not just rejects.
            for (int j=-8;j<width+8;++j)
                if (so+j>=0&&so+j<ns&&qo+j>=0&&qo+j<nq) q[qo+j]=s[so+j];
        }
        auto packed=pack(s), compressed=sliding(q);
        for (int w=1;w<=12 && w<=ns;++w) {
            int off=ns-w;
            Uint4 expected=0;
            for(int j=0;j<w;++j) expected=expected*4+s[off+j];
            assert(gpu_word(packed.data(),off,w)==expected);
        }
        BlastOffsetPair input={{Uint4(qo),Uint4(so)}}, actual={{0,0}};
        auto expected=reference(s,q,so,qo,width,word);
        kernel_s_BlastNaExtend_withoutHash_v3(packed.data(),q.data(),width,word-width,1,&input,&actual,ns,nq);
        assert(actual.qs_offsets.q_off==expected.qs_offsets.q_off);
        if (expected.qs_offsets.q_off!=Uint4(-1)) assert(actual.qs_offsets.s_off==expected.qs_offsets.s_off);
        GpuQueryContext context={0,nq};
        kernel_s_BlastSmallExtend_v3(packed.data(),compressed.data(),word,width,1,&input,&actual,ns,&context,0);
        assert(actual.qs_offsets.q_off==expected.qs_offsets.q_off);
        if (expected.qs_offsets.q_off!=Uint4(-1)) assert(actual.qs_offsets.s_off==expected.qs_offsets.s_off);
        ++checks;
    }
    for (int trial=0;trial<3000;++trial) {
        GpuQueryContext contexts[]={{0,17},{18,31},{50,33}};
        const auto& context=contexts[trial%3];
        std::vector<Uint1> q(83), subject(71);
        for(auto& b:q) b=rng()%4;
        for(auto& b:subject) b=rng()%4;
        int local=rng()%(context.query_length-8+1), so=rng()%(int(subject.size())-8+1);
        int qo=context.query_offset+local, word=9+rng()%7;
        for(int j=0;j<8;++j) q[qo+j]=subject[so+j];
        auto packed=pack(subject), compressed=sliding(q);
        std::vector<Uint1> context_query(q.begin()+context.query_offset,q.begin()+context.query_offset+context.query_length);
        auto expected=reference(subject,context_query,so,local,8,word);
        if (expected.qs_offsets.q_off!=Uint4(-1)) expected.qs_offsets.q_off+=context.query_offset;
        BlastOffsetPair input={{Uint4(qo),Uint4(so)}}, actual={{0,0}};
        kernel_s_BlastSmallExtend_v3(packed.data(),compressed.data(),word,8,1,&input,&actual,subject.size(),contexts,2);
        assert(actual.qs_offsets.q_off==expected.qs_offsets.q_off);
        if (expected.qs_offsets.q_off!=Uint4(-1)) assert(actual.qs_offsets.s_off==expected.qs_offsets.s_off);
    }
    std::printf("PASS: %u randomized tails/extensions, 3000 context boundaries, 100 scan-order checks and signed chain bounds (host emulation only)\n",checks);
}
