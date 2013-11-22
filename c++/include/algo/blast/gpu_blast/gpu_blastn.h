#ifndef __GPU_BLAST_H__
#define __GPU_BLAST_H__

#include <algo/blast/gpu_blast/gpu_blastn_config.hpp>
#include <algo/blast/core/blast_hspstream.h>
#include <algo/blast/gpu_blast/gpu_blast_multi_gpu_utils.hpp>
#include <algo/blast/gpu_blast/work_thread_base.hpp>
#include <algo/blast/gpu_blast/thread_work_queue.hpp>
#include <algo/blast/gpu_blast/work_thread.hpp>
#include <algo/blast/gpu_blast/utility.h>


/** Same as above, with support for user interrupt function
 * @param program_number Type of BLAST program [in]
 * @param query The query sequence [in]
 * @param query_info Additional query information [in]
 * @param seq_src Structure containing BLAST database [in]
 * @param score_options Hit scoring options [in]
 * @param sbp Scoring and statistical parameters [in]
 * @param lookup_wrap The lookup table, constructed earlier [in] 
 * @param word_options Options for processing initial word hits [in]
 * @param ext_options Options and parameters for the gapped extension [in]
 * @param hit_options Options for saving the HSPs [in]
 * @param eff_len_options Options for setting effective lengths [in]
 * @param psi_options Options specific to PSI-BLAST [in]
 * @param db_options Options for handling BLAST database [in]
 * @param hsp_stream Structure for streaming results [in] [out]
 * @param diagnostics Return statistics containing numbers of hits on 
 *                    different stages of the search [out]
 * @param interrupt_search User defined function to interrupt search [in]
 * @param progress_info User supplied data structure to aid interrupt [in]
 */
Int2 
	Blast_gpu_RunPreliminarySearchWithInterrupt(EBlastProgramType program, 
	BLAST_SequenceBlk* query, 
	BlastQueryInfo* query_info, 
	const BlastSeqSrc* seq_src, 
	const BlastScoringOptions* score_options,
	BlastScoreBlk* sbp, 
	LookupTableWrap* lookup_wrap,
	const BlastInitialWordOptions* word_options, 
	const BlastExtensionOptions* ext_options,
	const BlastHitSavingOptions* hit_options,
	const BlastEffectiveLengthsOptions* eff_len_options,
	const PSIBlastOptions* psi_options, 
	const BlastDatabaseOptions* db_options,
	const BlastGPUOptions* gpu_options,
	BlastHSPStream* hsp_stream, 
	BlastDiagnostics* diagnostics,
	TInterruptFnPtr interrupt_search, SBlastProgress* progress_info);

#endif //__GPU_BLAST_H__