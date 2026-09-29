// Real-device unit checks for the production CUDA kernel bodies. These are
// additional arithmetic/sanitizer tests, not full BLAST acceptance evidence.
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

typedef uint8_t Uint1;
typedef int16_t Int2;
typedef int32_t Int4;
typedef uint32_t Uint4;
struct BlastOffsetPair { struct { Uint4 q_off, s_off; } qs_offsets; };
struct GpuQueryContext { Int4 query_offset, query_length; };
#define COMPRESSION_RATIO 4
#define MIN(a,b) ((a)<(b)?(a):(b))
#include "../c++/src/algo/blast/gpu_blast/gpu_blastn_small_scan_kernel_v3.cuh"
#include "../c++/src/algo/blast/gpu_blast/gpu_blastn_mb_scan_kernel_v3.cuh"
#include "../c++/src/algo/blast/gpu_blast/gpu_blastn_lookup_hash_kernel_v3.cuh"
#include "../c++/src/algo/blast/gpu_blast/gpu_blastn_small_mini_extension_kernel_v3.cuh"
#include "../c++/src/algo/blast/gpu_blast/gpu_blastn_mini_extension_kernel_v3.cuh"

void check(cudaError_t status, const char* where)
{
    if (status!=cudaSuccess) {
        std::fprintf(stderr,"FAIL %s: %s\n",where,cudaGetErrorString(status));
        std::exit(1);
    }
}
#define CHECK(call) check(call,#call)
void require(bool ok, const char* reason)
{ if (!ok) { std::fprintf(stderr,"FAIL: %s\n",reason); std::exit(1); } }
template<class T> struct Buffer {
    T* p;
    explicit Buffer(size_t n) { CHECK(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T))); }
    explicit Buffer(const std::vector<T>& v) : Buffer(v.size()) { CHECK(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice)); }
    ~Buffer() { cudaFree(p); }
    std::vector<T> read(size_t n) const {
        std::vector<T> v(n);
        CHECK(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));
        return v;
    }
};
std::vector<Uint1> pack(const std::vector<Uint1>& sequence)
{
    std::vector<Uint1> result((sequence.size()+3)/4);
    for(size_t i=0;i<sequence.size();++i) result[i/4]|=sequence[i]<<(6-2*(i%4));
    return result;
}
std::vector<Uint1> sliding(const std::vector<Uint1>& sequence)
{
    std::vector<Uint1> result(sequence.size()+3);
    for(int i=-3;i<int(sequence.size());++i)
        for(int j=0;j<4;++j)
            if(i+j>=0&&i+j<int(sequence.size())) result[i+3]|=sequence[i+j]<<(6-2*j);
    return result;
}
Uint4 word_at(const std::vector<Uint1>& sequence, int offset, int width)
{
    Uint4 word=0;
    for(int j=0;j<width;++j) word=word*4+sequence[offset+j];
    return word;
}
BlastOffsetPair expected_extension(const std::vector<Uint1>& subject,
    const std::vector<Uint1>& query, BlastOffsetPair input, int width, int word)
{
    int s=input.qs_offsets.s_off,q=input.qs_offsets.q_off,left=0;
    while(left<word-width&&left<s&&left<q&&subject[s-left-1]==query[q-left-1]) ++left;
    int needed=word-width-left;
    if(s+width+needed>int(subject.size())||q+width+needed>int(query.size())) return {{Uint4(-1),0}};
    for(int r=0;r<needed;++r) if(subject[s+width+r]!=query[q+width+r]) return {{Uint4(-1),0}};
    return {{Uint4(q-left),Uint4(s-left)}};
}

template<bool Small> void run(const std::vector<Uint1>& subject,
    const std::vector<Uint1>& query, int width, int step, int start)
{
    size_t table_size=size_t(1)<<(2*width);
    std::vector<std::vector<Int4>> chains(table_size);
    for(int i=0;i<=int(query.size())-width;++i) chains[word_at(query,i,width)].push_back(i);
    std::vector<Int2> backbone(table_size,-1),overflow(2,-1);
    std::vector<Int4> hash(table_size,0),next(query.size()+1,0);
    Uint4 limit=1;
    for(size_t w=0;w<table_size;++w) {
        limit=std::max(limit,Uint4(chains[w].size()));
        if(chains[w].empty()) continue;
        hash[w]=chains[w][0]+1;
        for(size_t j=1;j<chains[w].size();++j) next[chains[w][j-1]+1]=chains[w][j]+1;
        if(chains[w].size()==1) backbone[w]=chains[w][0];
        else {
            require(overflow.size()<32768,"small overflow index fits Int2");
            backbone[w]=-Int2(overflow.size());
            for(auto q:chains[w]) overflow.push_back(q);
            overflow.push_back(-1);
        }
    }
    Uint4 positions=(subject.size()-width-start)/step+1;
    std::vector<BlastOffsetPair> expected;
    for(Uint4 i=0;i<positions;++i) {
        Uint4 s=start+i*step;
        for(auto q:chains[word_at(subject,s,width)]) expected.push_back({{Uint4(q),s}});
    }
    Buffer<Uint1> ds(pack(subject)),dq(Small?sliding(query):query);
    Buffer<Int2> db(backbone),doo(overflow);
    Buffer<Int4> dh(hash),dn(next);
    Buffer<Uint4> counts(positions+1),prefix(positions+1),error(1);
    Buffer<BlastOffsetPair> hits(size_t(positions)*limit),extended(std::max(expected.size(),size_t(1)));
    CHECK(cudaMemset(error.p,0,sizeof(Uint4)));
    CHECK(cudaMemset(counts.p+positions,0,sizeof(Uint4)));
    const Uint4 blocks=(positions+255)/256;
#define ARGS ds.p,start,step,width,positions,db.p,doo.p,Uint4(overflow.size()),dh.p,dn.p,Uint4(next.size()),limit,counts.p,prefix.p,hits.p,error.p
    gpu_scan_lookup<Small,false><<<blocks,256>>>(ARGS);
    CHECK(cudaGetLastError());
    thrust::device_ptr<Uint4> dc(counts.p),dp(prefix.p);
    thrust::exclusive_scan(dc,dc+positions+1,dp);
    auto offsets=prefix.read(positions+1);
    require(offsets.back()==expected.size(),"scan count matches independent sequence reference");
    gpu_scan_lookup<Small,true><<<blocks,256>>>(ARGS);
    CHECK(cudaGetLastError());
    auto actual=hits.read(expected.size());
    require(error.read(1)[0]==0,"bounded scan no chain error");
    for(size_t i=0;i<expected.size();++i) {
        require(actual[i].qs_offsets.q_off==expected[i].qs_offsets.q_off,"query-chain order");
        require(actual[i].qs_offsets.s_off==expected[i].qs_offsets.s_off,"subject scan order");
    }
#undef ARGS
    const int word=Small?11:28;
    require(!actual.empty(),"fixture must execute mini extension");
    if(Small) {
        Buffer<GpuQueryContext> contexts(std::vector<GpuQueryContext>{{0,Int4(query.size())}});
        if(step%4==0 && start%4==0)
            kernel_s_BlastSmallNaExtendAlignedOneByte<<<(actual.size()+255)/256,256>>>(ds.p,dq.p,word,width,
                actual.size(),hits.p,extended.p,subject.size(),contexts.p,0,query.size());
        else
            kernel_s_BlastSmallExtend_v3<<<(actual.size()+255)/256,256>>>(ds.p,dq.p,word,width,
                actual.size(),hits.p,extended.p,subject.size(),contexts.p,0);
        CHECK(cudaGetLastError());
        // The context buffer stays alive through this checked blocking copy.
        actual=extended.read(actual.size());
    } else {
        kernel_s_BlastNaExtend_withoutHash_v3<<<(actual.size()+255)/256,256>>>(ds.p,dq.p,width,
            word-width,actual.size(),hits.p,extended.p,subject.size(),query.size());
        CHECK(cudaGetLastError());
        actual=extended.read(actual.size());
    }
    size_t accepted=0;
    for(size_t i=0;i<expected.size();++i) {
        auto e=expected_extension(subject,query,expected[i],width,word);
        require(actual[i].qs_offsets.q_off==e.qs_offsets.q_off,"mini-extension query coordinate");
        if(e.qs_offsets.q_off!=Uint4(-1)) {
            require(actual[i].qs_offsets.s_off==e.qs_offsets.s_off,"mini-extension subject coordinate");
            ++accepted;
        }
    }
    require(accepted>0,"fixture must produce accepted mini extensions");
    std::printf("PASS real CUDA branch=%s width=%d step=%d start=%d scan=%zu extended=%zu\n",
        Small?"small":"megablast",width,step,start,expected.size(),accepted);
}
int main()
{
    cudaDeviceProp properties;
    CHECK(cudaGetDeviceProperties(&properties,0));
    require(properties.major==11&&properties.minor==0,"test requires Thor compute capability 11.0");
    std::printf("device=%s capability=%d.%d\n",properties.name,properties.major,properties.minor);
    std::mt19937 random(228);
    std::vector<Uint1> query(4003),subject(8003);
    for(auto& b:query) b=random()%4;
    for(auto& b:subject) b=random()%4;
    std::fill(query.begin()+1200,query.begin()+1270,0);
    std::fill(subject.begin()+5000,subject.begin()+5100,0);
    std::copy(query.begin()+100,query.begin()+220,subject.begin()+7213);
    std::copy(query.end()-29,query.end(),subject.end()-29);
    run<true>(subject,query,8,1,0);
    run<true>(subject,query,8,4,0);
    run<true>(subject,query,8,3,2);
    run<false>(subject,query,11,1,0);
    run<false>(subject,query,11,18,0);
    run<false>(subject,query,12,17,3);
    CHECK(cudaDeviceSynchronize());
    return 0;
}
