#include <algo/blast/core/na_ungapped.h>
#include <algo/blast/core/blast_nascan.h>
#include <algo/blast/core/blast_nalookup.h>
#include <algo/blast/gpu_blast/gpu_blastn_MB_and_smallNa.h>
#include <stdexcept>

void gpu_InitDBMemroy(int subject_seq_num, int max_len)
{ InitGPUMem_DB_MultiSeq(subject_seq_num, max_len); }
void gpu_ReleaseDBMemory() { ReleaseGPUMem_DB_MultiSeq(); }

void gpu_InitQueryMemory(LookupTableWrap* lookup, BLAST_SequenceBlk* query, BlastQueryInfo* info)
{
    if (lookup->lut_type == eSmallNaLookupTable) InitSmallQueryGPUMem(lookup, query, info);
    else if (lookup->lut_type == eMBLookupTable) InitMBQueryGPUMem(lookup, query);
    else throw std::runtime_error("G-BLASTN unsupported GPU nucleotide lookup type");
}
void gpu_ReleaseQueryMemory(LookupTableWrap* lookup)
{
    if (lookup->lut_type == eSmallNaLookupTable) ReleaseSmallQueryGPUMem();
    else if (lookup->lut_type == eMBLookupTable) ReleaseMBQueryGPUMem();
}
void GpuLookUpInit(LookupTableWrap* lookup)
{
    if (lookup->lut_type == eSmallNaLookupTable)
        lookup->lut = new BlastSmallNaLookupTable(*static_cast<BlastSmallNaLookupTable*>(lookup->lut));
    else if (lookup->lut_type == eMBLookupTable)
        lookup->lut = new BlastMBLookupTable(*static_cast<BlastMBLookupTable*>(lookup->lut));
    else throw std::runtime_error("G-BLASTN unsupported GPU nucleotide lookup type");
}
void GpuLookUpSetUp(LookupTableWrap* lookup)
{
    if (lookup->lut_type == eSmallNaLookupTable) {
        BlastSmallNaLookupTable* t = static_cast<BlastSmallNaLookupTable*>(lookup->lut);
        // The bounded scanner handles every small-table width and scan step.
        t->scansub_callback = reinterpret_cast<void*>(s_gpu_MBScanSubject_8_1Mod4_scankernel_Opt_v3);
        if (t->word_length == t->lut_word_length)
            t->extend_callback = reinterpret_cast<void*>(s_new_BlastNaExtendDirect);
        else if (t->lut_word_length % 4 == 0 && t->scan_step % 4 == 0 && t->word_length - t->lut_word_length <= 4)
            t->extend_callback = reinterpret_cast<void*>(s_gpu_BlastSmallNaExtendAlignedOneByte);
        else t->extend_callback = reinterpret_cast<void*>(s_gpu_BlastSmallExtend_v3);
    } else if (lookup->lut_type == eMBLookupTable) {
        BlastMBLookupTable* t = static_cast<BlastMBLookupTable*>(lookup->lut);
        if (t->discontiguous)
            throw std::runtime_error("Discontiguous megablast is not supported by the GPU backend; use -use_gpu false");
        t->scansub_callback = reinterpret_cast<void*>(s_gpu_MBScanSubject_Any_scankernel_Opt_v3);
        t->extend_callback = t->word_length == t->lut_word_length
            ? reinterpret_cast<void*>(s_new_BlastNaExtendDirect)
            : reinterpret_cast<void*>(s_gpu_BlastNaExtend_Opt_v3);
    } else throw std::runtime_error("G-BLASTN unsupported GPU nucleotide lookup type");
}

Int2 gpu_BlastNaWordFinder_v3(BLAST_SequenceBlk* subject, BLAST_SequenceBlk* query,
    BlastQueryInfo* info, LookupTableWrap* lookup, Int4** matrix,
    const BlastInitialWordParameters* params, Blast_ExtendWord* ewp,
    BlastOffsetPair* offsets, Int4 max_hits, BlastInitHitList* hits, BlastUngappedStats* stats)
{
    // Use the unchanged 2.2.28+ controller for mask ranges, callback dispatch,
    // diagonal lifetime and statistics. Its generic CPU fallback for subject
    // masking is intentional; the former local copies had no-op extenders.
    return BlastNaWordFinder(subject, query, info, lookup, matrix, params, ewp,
                             offsets, max_hits, hits, stats);
}
