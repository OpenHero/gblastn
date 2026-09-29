#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <algo/blast/core/blast_nalookup.h>
#include <algo/blast/core/blast_nascan.h>
#include <algo/blast/core/blast_util.h>
#include <algo/blast/core/blast_gapalign.h>
#include <algo/blast/gpu_blast/gpu_blastn_MB_and_smallNa.h>
#include <algo/blast/gpu_blast/gpu_blast_multi_gpu_utils.hpp>
#include "gpu_blastn_ungapped_extension_functions.h"

// Only these two fields are consumed by the device. Never memcpy the native
// BlastContextInfo (which contains Int8 and historically needed --no-align-double).
struct GpuQueryContext { Int4 query_offset; Int4 query_length; };
static_assert(sizeof(Int2) == 2 && Int2(-1) < 0, "signed 16-bit lookup required");
static_assert(sizeof(Int4) == 4 && sizeof(Uint4) == 4, "32-bit coordinates required");
static_assert(sizeof(GpuQueryContext) == 8 && alignof(GpuQueryContext) == 4,
              "query-context wire ABI");
static_assert(offsetof(GpuQueryContext, query_offset) == 0 &&
              offsetof(GpuQueryContext, query_length) == 4, "query-context offsets");
static_assert(sizeof(BlastOffsetPair) == 8 && alignof(BlastOffsetPair) == 4,
              "offset-pair wire ABI");

#include "gpu_blastn_small_scan_kernel_v3.cuh"
#include "gpu_blastn_mb_scan_kernel_v3.cuh"
#include "gpu_blastn_lookup_hash_kernel_v3.cuh"
#include "gpu_blastn_small_mini_extension_kernel_v3.cuh"
#include "gpu_blastn_mini_extension_kernel_v3.cuh"

namespace {
void Check(cudaError_t status, const char* operation)
{
    if (status != cudaSuccess) {
        std::ostringstream out;
        out << "G-BLASTN CUDA " << operation << ": " << cudaGetErrorString(status);
        throw std::runtime_error(out.str());
    }
}
#define GPU_CHECK(call) Check((call), #call)

bool Trace() { const char* v = std::getenv("GBLASTN_GPU_TRACE"); return v && v[0] == '1'; }
void Completed(const char* branch, const char* phase, int oid, size_t work, size_t hits)
{
    // Called only after a checked blocking D2H copy in the default stream.
    // Launch success or allocation alone is deliberately never called completion.
    if (Trace()) std::fprintf(stderr,
        "GBLASTN_KERNEL branch=%s phase=%s status=completed subject=%d work=%zu hits=%zu\n",
        branch, phase, oid, work, hits);
}

size_t AllocationLimit()
{
    const size_t mib = 1024u * 1024u;
    const char* value = std::getenv("GBLASTN_GPU_MEMORY_LIMIT_MB");
    if (!value || !*value) return 1024u * mib;
    char* end = NULL;
    errno = 0;
    unsigned long long n = std::strtoull(value, &end, 10);
    if (errno || *end || n < 16 || n > 16384)
        throw std::runtime_error("GBLASTN_GPU_MEMORY_LIMIT_MB must be 16..16384");
    return size_t(n) * mib;
}

// The process-wide budget includes all active device arrays (and resize peaks).
// Database subjects are never accumulated in a cache.
std::atomic<size_t> allocated_bytes(0), peak_bytes(0);
void ReserveBytes(size_t bytes)
{
    size_t prior = allocated_bytes.load();
    do {
        if (bytes > AllocationLimit() || prior > AllocationLimit() - bytes)
            throw std::runtime_error("G-BLASTN GPU memory budget exhausted; adjust GBLASTN_GPU_MEMORY_LIMIT_MB");
    } while (!allocated_bytes.compare_exchange_weak(prior, prior + bytes));
    size_t peak = peak_bytes.load();
    while (peak < prior + bytes && !peak_bytes.compare_exchange_weak(peak, prior + bytes)) {}
}
template<class T> class DeviceArray {
public:
    T* ptr;
    size_t size;
    DeviceArray() : ptr(NULL), size(0) {}
    ~DeviceArray() { if (ptr) { cudaFree(ptr); allocated_bytes.fetch_sub(size * sizeof(T)); } } // destructors never throw
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    void resize(size_t count) {
        if (count == size) return;
        if (count > AllocationLimit() / sizeof(T))
            throw std::runtime_error("G-BLASTN GPU allocation exceeds GBLASTN_GPU_MEMORY_LIMIT_MB");
        T* fresh = NULL;
        if (count) {
            ReserveBytes(count * sizeof(T));
            cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&fresh), count * sizeof(T));
            if (status != cudaSuccess) { allocated_bytes.fetch_sub(count * sizeof(T)); Check(status, "cudaMalloc"); }
        }
        if (ptr) {
            cudaError_t e = cudaFree(ptr);
            if (e != cudaSuccess) {
                if (fresh) { cudaFree(fresh); allocated_bytes.fetch_sub(count * sizeof(T)); }
                Check(e, "cudaFree");
            }
            allocated_bytes.fetch_sub(size * sizeof(T));
        }
        ptr = fresh; size = count;
    }
    void upload(const T* source, size_t count) {
        resize(count);
        if (count) GPU_CHECK(cudaMemcpy(ptr, source, count * sizeof(T), cudaMemcpyHostToDevice));
    }
};

struct ScanMemory : GpuObject {
    DeviceArray<Uint1> subject;
    DeviceArray<BlastOffsetPair> hits, extended;
    DeviceArray<Uint4> counts, prefix, error;
    int oid;
    Int4 length;
    Int8 cumulative_hits;
    const Uint1* source;
    ScanMemory() : oid(-1), length(-1), cumulative_hits(0), source(NULL) { error.resize(1); }
    void load(const BLAST_SequenceBlk* sequence) {
        if (sequence->length < 0 || !sequence->sequence)
            throw std::runtime_error("G-BLASTN invalid subject sequence");
        if (source == sequence->sequence && oid == sequence->oid && length == sequence->length) return;
        // ceil(length/4) retains partial final bytes. Four zero pad bytes support
        // the original mini-extension's bounded four-base comparisons at tails.
        size_t bytes = (size_t(sequence->length) + 3) / 4;
        subject.resize(bytes + 4);
        GPU_CHECK(cudaMemset(subject.ptr, 0, subject.size));
        if (bytes) GPU_CHECK(cudaMemcpy(subject.ptr, sequence->sequence, bytes, cudaMemcpyHostToDevice));
        source = sequence->sequence; oid = sequence->oid; length = sequence->length;
    }
};
struct QueryMemory : GpuObject {
    DeviceArray<Int2> backbone, overflow;
    DeviceArray<Int4> hash, next;
    DeviceArray<Uint1> query;
    DeviceArray<GpuQueryContext> contexts;
};
GpuData& Data() {
    GpuHandle* handle = BlastMGPUUtil.GetCurrentGPUHandle();
    if (!handle) throw std::runtime_error("G-BLASTN GPU callback has no device lease");
    return handle->Data;
}
ScanMemory& Scan() {
    if (!Data().m_global) throw std::runtime_error("G-BLASTN scan memory not initialized");
    return *static_cast<ScanMemory*>(Data().m_global);
}
QueryMemory& Query() {
    if (!Data().m_local) throw std::runtime_error("G-BLASTN query memory not initialized");
    return *static_cast<QueryMemory*>(Data().m_local);
}

__global__ void gpu_context_layout(Uint4* result, const GpuQueryContext* context,
                                   Int4 last)
{
    result[0] = sizeof(GpuQueryContext); result[1] = alignof(GpuQueryContext);
    result[2] = offsetof(GpuQueryContext, query_offset);
    result[3] = offsetof(GpuQueryContext, query_length);
    result[4] = sizeof(BlastContextInfo); result[5] = alignof(BlastContextInfo);
    result[6] = offsetof(BlastContextInfo, query_offset);
    result[7] = offsetof(BlastContextInfo, query_length);
    result[8] = context[0].query_offset; result[9] = context[0].query_length;
    result[10] = context[last].query_offset; result[11] = context[last].query_length;
}
void VerifyContextABI(const std::vector<GpuQueryContext>& host, QueryMemory& q)
{
    DeviceArray<Uint4> result; result.resize(12);
    gpu_context_layout<<<1,1>>>(result.ptr, q.contexts.ptr, Int4(host.size()-1));
    GPU_CHECK(cudaGetLastError());
    Uint4 layout[12];
    GPU_CHECK(cudaMemcpy(layout, result.ptr, sizeof(layout), cudaMemcpyDeviceToHost));
    if (layout[0] != sizeof(GpuQueryContext) || layout[1] != alignof(GpuQueryContext) ||
        layout[2] != 0 || layout[3] != 4 ||
        layout[8] != Uint4(host.front().query_offset) || layout[9] != Uint4(host.front().query_length) ||
        layout[10] != Uint4(host.back().query_offset) || layout[11] != Uint4(host.back().query_length))
        throw std::runtime_error("G-BLASTN host/device query-context ABI verification failed");
    if (Trace()) std::fprintf(stderr,
        "GBLASTN_ABI native_host_size=%zu native_host_align=%zu native_host_offsets=%zu,%zu "
        "native_device_size=%u native_device_align=%u native_device_offsets=%u,%u "
        "wire_size=%u wire_align=%u wire_offsets=%u,%u transferred_fields=verified\n",
        sizeof(BlastContextInfo), alignof(BlastContextInfo), offsetof(BlastContextInfo, query_offset),
        offsetof(BlastContextInfo, query_length), layout[4], layout[5], layout[6], layout[7],
        layout[0], layout[1], layout[2], layout[3]);
}

Int4 ScanSubject(const LookupTableWrap* lookup, const BLAST_SequenceBlk* subject,
                 BlastOffsetPair* output, Int4 max_hits, Int4* range)
{
    if (range[0] > range[1]) return 0;
    bool small = lookup->lut_type == eSmallNaLookupTable;
    Int4 step, width, chain;
    if (small) {
        const BlastSmallNaLookupTable* t = static_cast<const BlastSmallNaLookupTable*>(lookup->lut);
        step = t->scan_step; width = t->lut_word_length; chain = t->longest_chain;
    } else {
        const BlastMBLookupTable* t = static_cast<const BlastMBLookupTable*>(lookup->lut);
        if (t->discontiguous) throw std::runtime_error("Discontiguous megablast is not supported by the GPU backend");
        step = t->scan_step; width = t->lut_word_length; chain = t->longest_chain;
    }
    chain = std::max(chain, Int4(1));
    if (step < 1 || width < 1 || width > 12 || range[0] < 0 || range[1] > subject->length - width)
        throw std::runtime_error("G-BLASTN invalid GPU scan range or lookup width");
    if (max_hits < chain) throw std::runtime_error("G-BLASTN hit capacity is smaller than one lookup chain");
    ScanMemory& m = Scan(); QueryMemory& q = Query(); m.load(subject);
    if (range[0] == 0) m.cumulative_hits = 0;
    // At most 65536 scan positions and 1M hit slots per batch. Bound before
    // multiplication; all prefix sums fit Int4 and no hit is dropped on saturation.
    Uint4 capacity = std::min<Uint4>(Uint4(max_hits), 1024u * 1024u);
    if (Uint4(chain) > capacity) throw std::runtime_error("G-BLASTN lookup chain exceeds bounded GPU hit capacity");
    Uint4 positions = std::min<Uint4>(65536, capacity / Uint4(chain));
    positions = std::min<Uint4>(positions, Uint4((range[1] - range[0]) / step) + 1);
    m.counts.resize(positions + 1); m.prefix.resize(positions + 1);
    m.hits.resize(capacity);
    GPU_CHECK(cudaMemset(m.error.ptr, 0, sizeof(Uint4)));
    GPU_CHECK(cudaMemset(m.counts.ptr + positions, 0, sizeof(Uint4)));
    const Uint4 blocks = (positions + 255) / 256;
#define SCAN_ARGS m.subject.ptr, range[0], step, width, positions, q.backbone.ptr, q.overflow.ptr, Uint4(q.overflow.size), q.hash.ptr, q.next.ptr, Uint4(q.next.size), Uint4(chain), m.counts.ptr, m.prefix.ptr, m.hits.ptr, m.error.ptr
    if (small) gpu_scan_lookup<true,false><<<blocks,256>>>(SCAN_ARGS);
    else gpu_scan_lookup<false,false><<<blocks,256>>>(SCAN_ARGS);
    GPU_CHECK(cudaGetLastError());
    thrust::device_ptr<Uint4> counts(m.counts.ptr), prefix(m.prefix.ptr);
    thrust::exclusive_scan(counts, counts + positions + 1, prefix);
    GPU_CHECK(cudaGetLastError());
    Uint4 total = 0, error = 0;
    GPU_CHECK(cudaMemcpy(&total, m.prefix.ptr + positions, sizeof(total), cudaMemcpyDeviceToHost));
    GPU_CHECK(cudaMemcpy(&error, m.error.ptr, sizeof(error), cudaMemcpyDeviceToHost));
    if (error || total > capacity) throw std::runtime_error("G-BLASTN invalid or over-capacity lookup chain");
    m.cumulative_hits += total;
    if (m.cumulative_hits > INT_MAX)
        throw std::runtime_error("G-BLASTN subject hit count exceeds BLAST 2.2.28 signed 32-bit statistics capacity");
    if (small) gpu_scan_lookup<true,true><<<blocks,256>>>(SCAN_ARGS);
    else gpu_scan_lookup<false,true><<<blocks,256>>>(SCAN_ARGS);
    GPU_CHECK(cudaGetLastError());
    GPU_CHECK(cudaMemcpy(&error, m.error.ptr, sizeof(error), cudaMemcpyDeviceToHost));
    if (error) throw std::runtime_error("G-BLASTN lookup chain changed during scan");
    if (total) GPU_CHECK(cudaMemcpy(output, m.hits.ptr, size_t(total) * sizeof(*output), cudaMemcpyDeviceToHost));
#undef SCAN_ARGS
    Completed(small ? "small" : "megablast", "scan", subject->oid, positions, total);
    // The next offset can exceed INT_MAX if the final step crosses the end.
    Int8 next = Int8(range[0]) + Int8(positions) * step;
    range[0] = next > range[1] ? range[1] + 1 : Int4(next);
    return Int4(total);
}

Int4 Extend(BlastOffsetPair* offsets, Int4 count,
    const BlastInitialWordParameters* params, LookupTableWrap* lookup,
    BLAST_SequenceBlk* query, BLAST_SequenceBlk* subject, Int4** matrix,
    BlastQueryInfo* info, Blast_ExtendWord* ewp, BlastInitHitList* hitlist,
    Uint4 subject_range, bool aligned, bool direct)
{
    if (count <= 0) return 0;
    bool small = lookup->lut_type == eSmallNaLookupTable;
    Int4 word, width; BlastSeqLoc* masked;
    if (small) {
        BlastSmallNaLookupTable* t = static_cast<BlastSmallNaLookupTable*>(lookup->lut);
        word = t->word_length; width = t->lut_word_length; masked = t->masked_locations;
    } else {
        BlastMBLookupTable* t = static_cast<BlastMBLookupTable*>(lookup->lut);
        word = t->word_length; width = t->lut_word_length; masked = t->masked_locations;
    }
    ScanMemory& m = Scan(); QueryMemory& q = Query(); m.load(subject);
    if (size_t(count) > m.hits.size) throw std::runtime_error("G-BLASTN mini-extension input exceeds scan capacity");
    m.extended.resize(count);
    Int4 blocks = (count + 255) / 256;
    if (!direct) {
        if (small && aligned) {
            kernel_s_BlastSmallNaExtendAlignedOneByte<<<blocks,256>>>(m.subject.ptr, q.query.ptr,
                word, width, count, m.hits.ptr, m.extended.ptr, Int4(subject_range),
                q.contexts.ptr, info->last_context, query->length);
        } else if (small) {
            kernel_s_BlastSmallExtend_v3<<<blocks,256>>>(m.subject.ptr, q.query.ptr,
                word, width, count, m.hits.ptr, m.extended.ptr, Int4(subject_range),
                q.contexts.ptr, info->last_context);
        } else {
            kernel_s_BlastNaExtend_withoutHash_v3<<<blocks,256>>>(m.subject.ptr, q.query.ptr,
                width, word - width, count, m.hits.ptr, m.extended.ptr, Int4(subject_range), query->length);
        }
        GPU_CHECK(cudaGetLastError());
        GPU_CHECK(cudaMemcpy(offsets, m.extended.ptr, size_t(count) * sizeof(*offsets), cudaMemcpyDeviceToHost));
        size_t accepted = 0;
        for (Int4 i = 0; i < count; ++i) if (offsets[i].qs_offsets.q_off != Uint4(-1)) ++accepted;
        Completed(small ? "small" : "megablast", "extend", subject->oid, count, accepted);
    }
    Int4 extended = 0;
    // Keep the original CPU diagonal and scored ungapped extension. Input order
    // is the same as the CPU lookup scanner (including equal-subject ties).
    for (Int4 i = 0; i < count; ++i) {
        if (offsets[i].qs_offsets.q_off == Uint4(-1)) continue;
        Int4 s = offsets[i].qs_offsets.s_off, qoff = offsets[i].qs_offsets.q_off;
        if (params->container_type == eDiagHash)
            extended += s_BlastnDiagHashExtendInitialHit(query, subject, qoff, s,
                direct ? NULL : masked, info, subject_range, word, direct ? word : width,
                lookup, params, matrix, ewp->hash_table, hitlist);
        else
            extended += s_BlastnDiagTableExtendInitialHit(query, subject, qoff, s,
                direct ? NULL : masked, info, subject_range, word, direct ? word : width,
                lookup, params, matrix, ewp->diag_table, hitlist);
    }
    return extended;
}
} // namespace

void InitGPUMem_DB_MultiSeq(int, int)
{
    if (!Data().m_global) Data().m_global = new ScanMemory;
}
void ReleaseGPUMem_DB_MultiSeq()
{
    GpuData& data = Data(); delete static_cast<ScanMemory*>(data.m_global); data.m_global = NULL;
}
void InitSmallQueryGPUMem(LookupTableWrap* lookup, BLAST_SequenceBlk* query, BlastQueryInfo* info)
{
    if (Data().m_local) throw std::runtime_error("G-BLASTN query memory still in use");
    if (query->length < 0 || info->first_context < 0 || info->last_context < info->first_context)
        throw std::runtime_error("G-BLASTN invalid query context range");
    std::unique_ptr<QueryMemory> q(new QueryMemory);
    BlastSmallNaLookupTable* t = static_cast<BlastSmallNaLookupTable*>(lookup->lut);
    q->backbone.upload(t->final_backbone, t->backbone_size);
    q->overflow.upload(t->overflow, t->overflow_size);
    q->query.upload(query->compressed_nuc_seq_start, size_t(query->length) + 3);
    // Include contexts before first_context because lookup offsets and binary
    // search indexes remain relative to the original concatenated query.
    std::vector<GpuQueryContext> contexts(size_t(info->last_context) + 1);
    for (Int4 i = 0; i <= info->last_context; ++i) {
        contexts[i].query_offset = info->contexts[i].query_offset;
        contexts[i].query_length = info->contexts[i].query_length;
    }
    q->contexts.upload(&contexts[0], contexts.size());
    VerifyContextABI(contexts, *q);
    Scan().source = NULL;
    Data().m_local = q.release();
}
void InitMBQueryGPUMem(LookupTableWrap* lookup, BLAST_SequenceBlk* query)
{
    if (Data().m_local) throw std::runtime_error("G-BLASTN query memory still in use");
    BlastMBLookupTable* t = static_cast<BlastMBLookupTable*>(lookup->lut);
    if (t->discontiguous) throw std::runtime_error("Discontiguous megablast is not supported by the GPU backend");
    std::unique_ptr<QueryMemory> q(new QueryMemory);
    q->hash.upload(t->hashtable, t->hashsize);
    q->next.upload(t->next_pos, size_t(query->length) + 1);
    q->query.upload(query->sequence, query->length);
    Scan().source = NULL;
    Data().m_local = q.release();
}
void ReleaseSmallQueryGPUMem()
{
    GpuData& data = Data(); delete static_cast<QueryMemory*>(data.m_local); data.m_local = NULL;
    if (Trace()) std::fprintf(stderr, "GBLASTN_MEMORY allocated_bytes=%zu peak_bytes=%zu limit_bytes=%zu\n",
        allocated_bytes.load(), peak_bytes.load(), AllocationLimit());
}
void ReleaseMBQueryGPUMem() { ReleaseSmallQueryGPUMem(); }

#define DEFINE_SCAN(name) \
Int4 name(const LookupTableWrap* l, const BLAST_SequenceBlk* s, BlastOffsetPair* p, Int4 n, Int4* r) \
{ return ScanSubject(l, s, p, n, r); }
DEFINE_SCAN(s_gpu_MBScanSubject_8_1Mod4_scankernel_Opt_v3)
DEFINE_SCAN(s_gpu_BlastSmallNaScanSubject_8_4)
DEFINE_SCAN(s_gpu_BlastSmallNaScanSubject_8_4_v2)
DEFINE_SCAN(s_gpu_MBScanSubject_11_2Mod4_scankernel_Opt_v3)
DEFINE_SCAN(s_gpu_MBScanSubject_11_1Mod4_scankernel_Opt_v3)
DEFINE_SCAN(s_gpu_MBScanSubject_Any_scankernel_Opt_v3)
DEFINE_SCAN(s_gpu_MB_DiscWordScanSubject_11_18_1)
#undef DEFINE_SCAN
#define DEFINE_EXTEND(name, aligned, direct) \
Int4 name(const BlastOffsetPair* p, Int4 n, const BlastInitialWordParameters* w, LookupTableWrap* l, \
    BLAST_SequenceBlk* q, BLAST_SequenceBlk* s, Int4** m, BlastQueryInfo* i, \
    Blast_ExtendWord* e, BlastInitHitList* h, Uint4 r) \
{ return Extend(const_cast<BlastOffsetPair*>(p), n, w, l, q, s, m, i, e, h, r, aligned, direct); }
DEFINE_EXTEND(s_gpu_BlastSmallExtend_v3, false, false)
DEFINE_EXTEND(s_gpu_BlastSmallNaExtendAlignedOneByte, true, false)
DEFINE_EXTEND(s_gpu_BlastNaExtend_Opt_v3, false, false)
DEFINE_EXTEND(s_new_BlastNaExtendDirect, false, true)
#undef DEFINE_EXTEND
