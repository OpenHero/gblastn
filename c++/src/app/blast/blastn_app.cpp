/*  $Id: blastn_app.cpp 391262 2013-03-06 17:58:48Z rafanovi $
 * ===========================================================================
 *
 *                            PUBLIC DOMAIN NOTICE
 *               National Center for Biotechnology Information
 *
 *  This software/database is a "United States Government Work" under the
 *  terms of the United States Copyright Act.  It was written as part of
 *  the author's official duties as a United States Government employee and
 *  thus cannot be copyrighted.  This software/database is freely available
 *  to the public for use. The National Library of Medicine and the U.S.
 *  Government have not placed any restriction on its use or reproduction.
 *
 *  Although all reasonable efforts have been taken to ensure the accuracy
 *  and reliability of the software and data, the NLM and the U.S.
 *  Government do not and cannot warrant the performance or results that
 *  may be obtained by using this software or data. The NLM and the U.S.
 *  Government disclaim all warranties, express or implied, including
 *  warranties of performance, merchantability or fitness for any particular
 *  purpose.
 *
 *  Please cite the author in any work or product based on this material.
 *
 * ===========================================================================
 *
 * Authors:  Christiam Camacho
 *
 */

/** @file blastn_app.cpp
 * BLASTN command line application
 */

#ifndef SKIP_DOXYGEN_PROCESSING
static char const rcsid[] = 
	"$Id: blastn_app.cpp 391262 2013-03-06 17:58:48Z rafanovi $";
#endif /* SKIP_DOXYGEN_PROCESSING */

#include <ncbi_pch.hpp>
#include <corelib/ncbiapp.hpp>
#include <algo/blast/api/local_blast.hpp>
#include <algo/blast/api/blast_nucl_options.hpp>
#include <algo/blast/api/remote_blast.hpp>
#include <algo/blast/blastinput/blast_fasta_input.hpp>
#include <algo/blast/blastinput/blastn_args.hpp>
#include <algo/blast/api/objmgr_query_data.hpp>
#include <algo/blast/format/blast_format.hpp>
#include "blast_app_util.hpp"

#include <corelib/ncbifile.hpp>
#include <algo/blast/gpu_blast/gpu_blastn.h>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <future>
#include <memory>
#include <set>
#include <stdexcept>

USING_NCBI_SCOPE;
USING_SCOPE(blast);
USING_SCOPE(objects);

namespace {
// Preliminary search temporarily changes effective search spaces when splitting
// queries. Every in-flight batch needs its own options, even within one job.
class CBatchOptions : public CBlastNucleotideOptionsHandle {
public:
    explicit CBatchOptions(const CBlastOptions& source)
        : CBlastNucleotideOptionsHandle(source.GetLocality()) {
        m_Opts = source.Clone();
    }
};

// Jobs own their arguments (and therefore streams), options, scopes and
// formatters until every submitted batch has been formatted. No stream or
// formatter is shared across query-list entries.
struct SQueryJob {
    std::unique_ptr<CArgs> args;
    CRef<CBlastnAppArgs> command;
    CRef<CBlastOptionsHandle> options;
    CRef<CBlastDatabaseArgs> database;
    CRef<CLocalDbAdapter> adapter;
    CRef<CScope> scope;
    std::unique_ptr<CBlastFormat> formatter;
    string name;
    size_t submitted = 0;
    size_t completed = 0;
    bool sealed = false;
    bool finished = false;

    void FinishIfReady() {
        if (!sealed || finished || completed != submitted) return;
        formatter->PrintEpilog(options->GetOptions());
        command->GetOutputStream().flush();
        if (!command->GetOutputStream())
            throw std::runtime_error("G-BLASTN: failed writing output for " + name);
        if (command->ProduceDebugOutput())
            options->GetOptions().DebugDumpText(NcbiCerr, "BLAST options", 1);
        finished = true;
    }
};

bool BatchTrace()
{
    const char* pipeline = std::getenv("GBLASTN_PIPELINE_TRACE");
    const char* gpu = std::getenv("GBLASTN_GPU_TRACE");
    return (pipeline && pipeline[0] == '1') || (gpu && gpu[0] == '1');
}

struct SQueryBatch {
    std::shared_ptr<SQueryJob> job;
    size_t index = 0;
    CRef<CBlastQueryVector> query;
    CRef<IQueryFactory> factory;
    CRef<CBlastOptionsHandle> options;
    std::unique_ptr<CLocalBlast> search;
    CRef<CSearchResultSet> results;
    bool needs_traceback = false;
};
typedef std::shared_ptr<SQueryBatch> TBatch;

void FormatBatch(const TBatch& batch)
{
    SQueryJob& job = *batch->job;
    if (batch->results.Empty())
        throw std::runtime_error("G-BLASTN: no search results for " + job.name);
    ITERATE(CSearchResultSet, result, *batch->results) {
        if ((**result).HasErrors())
            throw std::runtime_error("G-BLASTN: " + (**result).GetErrorStrings());
    }
    if (job.command->GetFormattingArgs()->ArchiveFormatRequested(*job.args)) {
        job.formatter->WriteArchive(*batch->factory, *batch->options, *batch->results);
    } else {
        BlastFormatter_PreFetchSequenceData(*batch->results, job.scope);
        ITERATE(CSearchResultSet, result, *batch->results)
            job.formatter->PrintOneResultSet(**result, batch->query);
    }
    ++job.completed;
    if (BatchTrace())
        std::fprintf(stderr, "GBLASTN_BATCH query=\"%s\" index=%zu stage=completed\n",
                     job.name.c_str(), batch->index);
    job.FinishIfReady();
}

// At most one preliminary search and one traceback run concurrently. This
// bounds live batches and GPU memory, and avoids concurrent use of the same
// device's lookup buffers. Formatting is ordered on the calling thread.
// Future destruction joins outstanding tasks, including exception paths.
class CQueryPipeline {
public:
    void Submit(const TBatch& batch) {
        AdvancePreliminary();
        m_Preliminary = std::async(std::launch::async, [batch]() {
            const int status = batch->search->PrelimSearchRun(batch->results);
            batch->needs_traceback = status == 0;
            if (status != 0 && batch->results.Empty())
                throw std::runtime_error("G-BLASTN: preliminary search failed");
            return batch;
        });
    }
    void Drain() {
        AdvancePreliminary();
        if (m_Traceback.valid()) FormatBatch(m_Traceback.get());
    }
private:
    void AdvancePreliminary() {
        if (!m_Preliminary.valid()) return;
        TBatch previous = m_Preliminary.get();
        if (m_Traceback.valid()) FormatBatch(m_Traceback.get());
        m_Traceback = std::async(std::launch::async, [previous]() {
            if (previous->needs_traceback)
                previous->results = previous->search->TraceBackRun();
            return previous;
        });
    }
    std::future<TBatch> m_Preliminary;
    std::future<TBatch> m_Traceback;
};

// Constructed before the pipeline, so all workers finish before GPU release.
struct SGpuSession {
    ~SGpuSession() { Blast_gpu_Release(); }
};

struct SQueryPath { string input; string output; };

string ResolvedPath(const string& path)
{
    return CDirEntry::NormalizePath(CDirEntry::CreateAbsolutePath(path), eFollowLinks);
}

vector<SQueryPath> ReadQueryList(const string& filename, const string& output)
{
    std::ifstream list(filename.c_str());
    if (!list)
        throw std::runtime_error("G-BLASTN: cannot open query list: " + filename);
    vector<SQueryPath> paths;
    std::set<string> outputs;
    std::set<string> inputs;
    const string::size_type slash = output.find_last_of("/\\");
    const string directory = slash == string::npos ? "" : output.substr(0, slash + 1);
    string line;
    while (std::getline(list, line)) {
        line = NStr::TruncateSpaces(line);
        if (line.empty()) continue;
        std::ifstream query(line.c_str());
        if (!query || query.peek() == std::ifstream::traits_type::eof())
            throw std::runtime_error("G-BLASTN: missing or empty query-list input: " + line);
        const string::size_type pos = line.find_last_of("/\\");
        const string name = pos == string::npos ? line : line.substr(pos + 1);
        SQueryPath path = {line, directory + name + ".out"};
        const string absolute_output = ResolvedPath(path.output);
        if (!outputs.insert(absolute_output).second)
            throw std::runtime_error("G-BLASTN: query-list output-name collision: " + path.output);
        inputs.insert(ResolvedPath(line));
        paths.push_back(path);
    }
    if (list.bad()) throw std::runtime_error("G-BLASTN: error reading query list: " + filename);
    if (paths.empty()) throw std::runtime_error("G-BLASTN: query list contains no inputs: " + filename);
    // SetOptions opens the command's -out stream before per-entry streams.
    const string initial_output = ResolvedPath(output);
    if (output != "-" && (inputs.count(initial_output) ||
        initial_output == ResolvedPath(filename)))
        throw std::runtime_error("G-BLASTN: -out would overwrite a query-list input: " + output);
    for (std::set<string>::const_iterator out = outputs.begin(); out != outputs.end(); ++out)
        if (inputs.count(*out) || *out == ResolvedPath(filename))
            throw std::runtime_error("G-BLASTN: query-list output would overwrite an input: " + *out);
    return paths;
}
}

class CBlastnApp : public CNcbiApplication {
public:
    CBlastnApp() {
        CRef<CVersion> version(new CVersion());
        version->SetVersionInfo(new CBlastVersion());
        SetFullVersion(version);
    }
private:
    void Init() {
        m_Command.Reset(new CBlastnAppArgs());
        HideStdArgs(fHideLogfile | fHideConffile | fHideFullVersion | fHideXmlHelp | fHideDryRun);
        SetupArgDescriptions(m_Command->SetCommandLine());
    }
    int Run();
    void ProcessJob(const SQueryPath* path, int mode, CQueryPipeline& pipeline);
    CRef<CBlastnAppArgs> m_Command;
    CRef<CBlastOptionsHandle> m_Options;
    bool m_IndexInitialized = false;
    bool m_IndexLoaded = false;
    ELookupTableType m_IndexLookupType = eMBLookupTable;
};

void CBlastnApp::ProcessJob(const SQueryPath* path, int mode, CQueryPipeline& pipeline)
{
    std::shared_ptr<SQueryJob> job(new SQueryJob());
    job->args.reset(new CArgs(GetArgs()));
    job->command = path ? CRef<CBlastnAppArgs>(new CBlastnAppArgs()) : m_Command;
    if (path) {
        std::unique_ptr<CArgDescriptions> descriptions(job->command->SetCommandLine());
        unsigned int plain = kMax_UInt;
        descriptions->x_CreateArg("-query", true, path->input, &plain, *job->args, true);
        descriptions->x_CreateArg("-out", true, path->output, &plain, *job->args, true);
    }
    job->name = path ? path->input : (*job->args)[kArgQuery].AsString();
    if (!path) {
        // Reuse the already parsed command so imported strategies and stdin
        // are consumed exactly once.
        job->options = m_Options;
    } else {
        job->options.Reset(&*job->command->SetOptions(*job->args));
    }
    const CBlastOptions& opt = job->options->GetOptions();
    job->database = job->command->GetBlastDatabaseArgs();
    InitializeSubject(job->database, job->options, job->command->ExecuteRemotely(),
                      job->adapter, job->scope);
    CRef<CQueryOptionsArgs> query_options = job->command->GetQueryOptionsArgs();
    SDataLoaderConfig loaders = InitializeQueryDataLoaderConfiguration(
        query_options->QueryIsProtein(), job->adapter);
    CBlastInputSourceConfig config(loaders, query_options->GetStrand(),
        query_options->UseLowercaseMasks(), query_options->GetParseDeflines(),
        query_options->GetRange());
    config.SetQueryLocalIdMode();
    CBlastFastaInputSource fasta(job->command->GetInputStream(), config);
    const int batch_size = job->command->GetQueryBatchSize();
    if (batch_size <= 0)
        throw std::runtime_error("G-BLASTN: BATCH_SIZE must be a positive integer");
    CBlastInput input(&fasta, batch_size);
    if (opt.GetUseIndex() && !job->command->ExecuteRemotely()) {
        // The index callbacks are process-wide. Initialize once, before any
        // submitted search, rather than replacing them under an active job.
        if (!m_IndexInitialized) {
            CRef<CBlastOptions> options(&job->options->SetOptions());
            CSetupFactory::InitializeMegablastDbIndex(options);
            m_IndexLoaded = opt.GetMBIndexLoaded();
            m_IndexLookupType = opt.GetLookupTableType();
            m_IndexInitialized = true;
        } else {
            job->options->SetOptions().SetMBIndexLoaded(m_IndexLoaded);
            if (m_IndexLoaded)
                job->options->SetOptions().SetLookupTableType(m_IndexLookupType);
            else
                job->options->SetOptions().SetUseIndex(false);
        }
        if (m_IndexLoaded && opt.GetUseGpu())
            throw std::runtime_error("G-BLASTN: indexed megablast bypasses CUDA scanning; use -use_index false with -use_gpu true");
    }
    CRef<CFormattingArgs> format = job->command->GetFormattingArgs();
    job->formatter.reset(new CBlastFormat(opt, *job->adapter,
        format->GetFormattedOutputChoice(), query_options->GetParseDeflines(),
        job->command->GetOutputStream(), format->GetNumDescriptions(),
        format->GetNumAlignments(), *job->scope, opt.GetMatrixName(),
        format->ShowGis(), format->DisplayHtmlOutput(), opt.GetQueryGeneticCode(),
        opt.GetDbGeneticCode(), opt.GetSumStatisticsMode(),
        job->command->ExecuteRemotely(), job->adapter->GetFilteringAlgorithm(),
        format->GetCustomOutputFormatSpec(), job->command->GetTask() == "megablast",
        opt.GetMBIndexLoaded()));
    job->formatter->PrintProlog();
    CBatchSizeMixer mixer(2000000, SplitQuery_GetChunkSize(opt.GetProgram()) - 1000);
    const char* requested_batch_size = std::getenv("BATCH_SIZE");
    const bool fixed_batch_size = requested_batch_size && *requested_batch_size;
    if (!job->command->ExecuteRemotely() && !fixed_batch_size)
        input.SetBatchSize(mixer.GetBatchSize());

    for (; !input.End();) {
        TBatch batch(new SQueryBatch());
        batch->job = job;
        batch->query.Reset(input.GetNextSeqBatch(*job->scope));
        if (batch->query->Empty()) continue;
        batch->factory.Reset(new CObjMgr_QueryFactory(*batch->query));
        batch->options.Reset(new CBatchOptions(opt));
        SaveSearchStrategy(*job->args, job->command, batch->factory, batch->options);
        batch->index = ++job->submitted;
        if (BatchTrace())
            std::fprintf(stderr, "GBLASTN_BATCH query=\"%s\" index=%zu stage=submitted\n",
                         job->name.c_str(), batch->index);
        if (job->command->ExecuteRemotely()) {
            CRef<CRemoteBlast> remote = InitializeRemoteBlast(batch->factory,
                job->database, batch->options, job->command->ProduceDebugRemoteOutput(),
                job->command->GetClientId());
            batch->results = remote->GetResultSet();
            FormatBatch(batch);
        } else {
            batch->search.reset(new CLocalBlast(batch->factory, batch->options, job->adapter));
            batch->search->SetNumberOfThreads(job->command->GetNumThreads());
            if (mode == 1) {
                batch->results = batch->search->Run();
                if (!fixed_batch_size)
                    input.SetBatchSize(mixer.GetBatchSize(batch->search->GetNumExtensions()));
                FormatBatch(batch);
                job->formatter->ResetScopeHistory();
            } else {
                pipeline.Submit(batch);
            }
        }
    }
    if (job->submitted == 0)
        throw std::runtime_error("G-BLASTN: query contains no usable sequences: " + job->name);
    job->sealed = true;
    job->FinishIfReady();
    if (mode == 0) pipeline.Drain();
}

int CBlastnApp::Run()
{
    int status = BLAST_EXIT_SUCCESS;
    const std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
    try {
        SetDiagPostLevel(eDiag_Warning);
        const CArgs& args = GetArgs();
        const bool required = args["gpu_required"].AsBoolean();
        if (required && !args[kArgUseGpu].AsBoolean())
            throw std::runtime_error("G-BLASTN: -gpu_required true requires -use_gpu true");
        vector<SQueryPath> paths;
        if (!args[kArgQueryList].AsString().empty()) {
            paths = ReadQueryList(args[kArgQueryList].AsString(), args[kArgOutput].AsString());
            if (args.Exist(kArgInputSearchStrategy) && args[kArgInputSearchStrategy].HasValue())
                throw std::runtime_error("G-BLASTN: query_list cannot be combined with an imported search strategy");
        } else if (args[kArgQuery].AsString() != "-" && args[kArgOutput].AsString() != "-" &&
                   ResolvedPath(args[kArgQuery].AsString()) == ResolvedPath(args[kArgOutput].AsString())) {
            throw std::runtime_error("G-BLASTN: -out would overwrite the query input");
        }
        CRef<CBlastOptionsHandle> options;
        if (RecoverSearchStrategy(args, m_Command))
            options.Reset(&*m_Command->SetOptionsForSavedStrategy(args));
        else
            options.Reset(&*m_Command->SetOptions(args));
        // Execution settings are never recovered from an algorithm strategy.
        CGpuArgs execution_options;
        execution_options.ExtractAlgorithmOptions(args, options->SetOptions());
        m_Options = options;
        const CBlastOptions& opt = options->GetOptions();
        int mode = opt.GetMethod();
        if (mode < 0 || mode > 3) throw std::runtime_error("G-BLASTN: invalid -mode; expected 0, 1, 2 or 3");
        if (mode == 3) mode = 2;
        if (opt.GetUseGpu() && m_Command->ExecuteRemotely())
            throw std::runtime_error("G-BLASTN: -use_gpu true requires a local search");
        if (opt.GetUseGpu() && m_Command->GetTask() != "blastn" && m_Command->GetTask() != "megablast")
            throw std::runtime_error("G-BLASTN: GPU execution supports blastn and contiguous megablast only");
        if (opt.GetConverted())
            throw std::runtime_error("G-BLASTN: converted-database coordinate remapping is unsupported; use an ordinary v4 BLAST database");
        if (mode != 1 && (opt.GetPrepareNum() != 1 || opt.GetPrelimNum() != 1 ||
                          opt.GetTraceNum() != 1 || opt.GetPrintNum() != 1))
            throw std::runtime_error("G-BLASTN: bounded pipeline requires one prepare, preliminary, traceback and print stage; use -num_threads for search parallelism");
        SGpuSession gpu;
        Blast_gpu_Init(opt.GetUseGpu(), opt.GetGpuID());
        std::cerr << "GBLASTN_RUN mode=" << mode << " gpu_required=" << (required ? "true" : "false") << '\n';
        CQueryPipeline pipeline;
        if (paths.empty()) {
            ProcessJob(NULL, mode, pipeline);
        } else {
            for (size_t i = 0; i < paths.size(); ++i) {
                std::fprintf(stderr, "GBLASTN_QUERY input=\"%s\" output=\"%s\"\n",
                             paths[i].input.c_str(), paths[i].output.c_str());
                ProcessJob(&paths[i], mode, pipeline);
            }
        }
        pipeline.Drain();
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::cerr << "GBLASTN_RUN status=completed elapsed_seconds=" << seconds << '\n';
    } CATCH_ALL(status)
    return status;
}

#ifndef SKIP_DOXYGEN_PROCESSING
int main(int argc, const char* argv[])
{
    return CBlastnApp().AppMain(argc, argv, 0, eDS_Default, 0);
}
#endif
