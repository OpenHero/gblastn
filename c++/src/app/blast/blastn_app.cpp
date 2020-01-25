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
#include <algo/blast/api/remote_blast.hpp>
#include <algo/blast/blastinput/blast_fasta_input.hpp>
#include <algo/blast/blastinput/blastn_args.hpp>
#include <algo/blast/api/objmgr_query_data.hpp>
#include <algo/blast/format/blast_format.hpp>
#include "blast_app_util.hpp"

///// CHANGED by kyzhao /////
//#include "utility.h"
#include <sstream>
#if __linux__
#include <unistd.h>
#endif

//#include <string>
///// CHANGED /////

//////////////////////////////////////////////////////////////////////////
// kyzhao for gpu
//#include <windows.h>
#include <algo/blast/gpu_blast/gpu_logfile.h>
#include <algo/blast/gpu_blast/gpu_blastn.h>

#ifndef SKIP_DOXYGEN_PROCESSING
USING_NCBI_SCOPE;
USING_SCOPE(blast);
USING_SCOPE(objects);
#endif

class CBlastnApp : public CNcbiApplication
{
public:
    /** @inheritDoc */
    CBlastnApp() {
        CRef<CVersion> version(new CVersion());
        version->SetVersionInfo(new CBlastVersion());
        SetFullVersion(version);

		//////////////////////////////////////////////////////////////////////////
		job_queue = NULL;
		query_queue = NULL;
		prelim_queue = NULL;
		result_queue = NULL;

		prelim_thread = NULL;
		trace_thread = NULL;
		print_thread = NULL;
		rdsmap = NULL;
		//////////////////////////////////////////////////////////////////////////
		
    }

	//////////////////////////////////////////////////////////////////////////
	//added by kyzhao
	~CBlastnApp();
	//////////////////////////////////////////////////////////////////////////
	
private:
    /** @inheritDoc */
    virtual void Init();
    /** @inheritDoc */
    virtual int Run();

	//////////////////////////////////////////////////////////////////////////
	//added by kyzhao
	int Method1(const CBlastOptions& opt);
	int Method2(const CBlastOptions& opt);
	int Method3(const CBlastOptions& opt);

	int Method1_opt(string job_name);
	int Method2_opt(string job_name);
	int Method3_opt(string job_name);

	//void FormatResult(string outfile);
	//////////////////////////////////////////////////////////////////////////
	work_queue<JobItem*> * job_queue;// = new work_queue<JobItem*>;
	work_queue<work_item*> * query_queue;// = new work_queue<work_item*>;
	work_queue<work_item*> * prelim_queue;// = new work_queue<work_item*>;
	work_queue<work_item*> * result_queue;// = new work_queue<work_item*>;

	PrelimSearchThread* prelim_thread;// = new PrelimSearchThread[prelim_num];
	TraceBackThread* trace_thread;// = new TraceBackThread[trace_num];
	WorkThreadBase* print_thread;// = new PrintThread[print_num];

	string output;
	NewRecordsMap* rdsmap;
	//////////////////////////////////////////////////////////////////////////
	
    /// This application's command line args
    CRef<CBlastnAppArgs> m_CmdLineArgs; 


};

//////////////////////////////////////////////////////////////////////////

CBlastnApp::~CBlastnApp()
{
	if (job_queue != NULL)
	{
		delete job_queue;
	}
	if (query_queue != NULL)
	{
		delete query_queue;
	}
	if (prelim_queue != NULL)
	{
		delete prelim_queue;
	}
	if (result_queue != NULL)
	{
		delete result_queue;
	}

	if (prelim_queue != NULL)
	{
		delete[] prelim_thread;
	}

	if (trace_thread != NULL)
	{
		delete[] trace_thread;
	}
	if (print_thread != NULL)
	{
		delete[] print_thread;
	}
	if (rdsmap != NULL)
	{
		delete rdsmap;
	}
}
//////////////////////////////////////////////////////////////////////////

//void CBlastnApp::FormatResult(string outfile)
//{
//	if (rdsmap == NULL)
//	{
//		printf("Can't load record map!\n");
//		//exit(-1);
//	}else
//	{
//		///// CHANGED /////
//		//string outputfile = args[kArgOutput].AsString();
//		//printf("Output file : %s\n", outputfile.c_str());
//		//printf("Output Choice : %d\n", m_CmdLineArgs->GetFormattingArgs()->GetFormattedOutputChoice());
//		if ( m_CmdLineArgs->GetFormattingArgs()->GetFormattedOutputChoice() == 7){//CFormattingArgs::EOutputFormat::eTabularWithComments ) {
//			// Flush all content first
//			//CNcbiIostream& ioStream = m_CmdLineArgs->GetInputOutputStream();
//			//ioStream.flush();
//			cout << "outfile" <<outfile<<endl;
//			fstream ioStream;
//			ioStream.open(outfile.c_str());
//
//			// Return to the beginning
//			//ioStream.seekg(0);
//
//			std::vector<std::string> lines;
//			std::string line;
//			int line_count = 0;
//			while (std::getline(ioStream, line))
//			{
//				lines.push_back(line);
//			}
//			printf("Line Count : %d\n", lines.size());
//
//			// Return to the beginning
//			ioStream.clear();
//			ioStream.seekg(0);
//
//			for ( int i = 0; i < lines.size(); i++ ) {
//				if ( lines[i][0] == '#' ) {
//					ioStream.write(lines[i].c_str(), lines[i].length());
//					ioStream.write("\n", 1);
//					//fprintf(ioStream, "%s", line);
//					//ioStream << lines[i] <<"\r";
//				}
//				else {
//					std::vector<std::string> tokens;
//					char* pch = strtok((char*)lines[i].c_str(),"\t");
//
//					int k = 0;
//					while ( pch != NULL ) {
//						tokens.push_back(std::string(pch));
//						pch = strtok (NULL, "\t");
//						k++;
//					}
//
//					//cout << "k:"<< k<<endl;
//					stringstream myStrStart(tokens[8]);
//					stringstream myStrEnd(tokens[9]);
//					unsigned int start_offset;// = atoi(tokens[8].c_str());
//					unsigned int end_offset;// = atoi(tokens[9].c_str());
//					myStrStart >> start_offset;
//					myStrEnd >> end_offset;
//
//					NewRecord* found = rdsmap->getCorrectedRecord(tokens[1], start_offset, end_offset);
//					if ( found != NULL ) {
//						string buf;
//						char start_offset_s[50];
//						char end_offset_s[50];
//						sprintf(start_offset_s, "%d", start_offset);
//						sprintf(end_offset_s, "%d", end_offset);
//
//						buf.append(tokens[0]).append("\t")
//							.append(found->_id).append("\t")
//							.append(tokens[2]).append("\t")
//							.append(tokens[3]).append("\t")
//							.append(tokens[4]).append("\t")
//							.append(tokens[5]).append("\t")
//							.append(tokens[6]).append("\t")
//							.append(tokens[7]).append("\t")
//							.append(start_offset_s).append("\t")
//							.append(end_offset_s).append("\t")
//							.append(tokens[10]).append("\t")
//							.append(tokens[11]).append("\n");
//						ioStream.write(buf.c_str(), buf.length());
//						//fprintf(fo, "%s", buf.c_str());
//					}
//					else
//						continue;
//				}
//			}
//
//			ioStream.close();
//		}
//		printf("Complete Correcting!\n");
//		///// CHANGED /////
//	}
//}

void CBlastnApp::Init()
{
    // formulate command line arguments

    m_CmdLineArgs.Reset(new CBlastnAppArgs());

    // read the command line

    HideStdArgs(fHideLogfile | fHideConffile | fHideFullVersion | fHideXmlHelp | fHideDryRun);
    SetupArgDescriptions(m_CmdLineArgs->SetCommandLine());
}

#if 0
int CBlastnApp::Run(void)
{
    int status = BLAST_EXIT_SUCCESS;

    try {

        // Allow the fasta reader to complain on invalid sequence input
        SetDiagPostLevel(eDiag_Warning);

        /*** Get the BLAST options ***/
        const CArgs& args = GetArgs();
        CRef<CBlastOptionsHandle> opts_hndl;
        if(RecoverSearchStrategy(args, m_CmdLineArgs)){
        	opts_hndl.Reset(&*m_CmdLineArgs->SetOptionsForSavedStrategy(args));
        }
        else {
        	opts_hndl.Reset(&*m_CmdLineArgs->SetOptions(args));
        }
        const CBlastOptions& opt = opts_hndl->GetOptions();

        /*** Initialize the database/subject ***/
        CRef<CBlastDatabaseArgs> db_args(m_CmdLineArgs->GetBlastDatabaseArgs());
        CRef<CLocalDbAdapter> db_adapter;
        CRef<CScope> scope;
        InitializeSubject(db_args, opts_hndl, m_CmdLineArgs->ExecuteRemotely(),
                         db_adapter, scope);
        _ASSERT(db_adapter && scope);

        /*** Get the query sequence(s) ***/
        CRef<CQueryOptionsArgs> query_opts = 
            m_CmdLineArgs->GetQueryOptionsArgs();
        SDataLoaderConfig dlconfig =
            InitializeQueryDataLoaderConfiguration(query_opts->QueryIsProtein(),
                                                   db_adapter);
        CBlastInputSourceConfig iconfig(dlconfig, query_opts->GetStrand(),
                                     query_opts->UseLowercaseMasks(),
                                     query_opts->GetParseDeflines(),
                                     query_opts->GetRange());
        iconfig.SetQueryLocalIdMode();
        CBlastFastaInputSource fasta(m_CmdLineArgs->GetInputStream(), iconfig);
        CBlastInput input(&fasta, m_CmdLineArgs->GetQueryBatchSize());

        // Initialize the megablast database index now so we can know whether an indexed search will be run.
        // This is only important for the reference in the report, but would be done anyway.
        if (opt.GetUseIndex() && !m_CmdLineArgs->ExecuteRemotely()) {
            CRef<CBlastOptions> my_options(&(opts_hndl->SetOptions()));
            CSetupFactory::InitializeMegablastDbIndex(my_options);
        }

        /*** Get the formatting options ***/
        CRef<CFormattingArgs> fmt_args(m_CmdLineArgs->GetFormattingArgs());
        CBlastFormat formatter(opt, *db_adapter,
                               fmt_args->GetFormattedOutputChoice(),
                               query_opts->GetParseDeflines(),
                               m_CmdLineArgs->GetOutputStream(),
                               fmt_args->GetNumDescriptions(),
                               fmt_args->GetNumAlignments(),
                               *scope,
                               opt.GetMatrixName(),
                               fmt_args->ShowGis(),
                               fmt_args->DisplayHtmlOutput(),
                               opt.GetQueryGeneticCode(),
                               opt.GetDbGeneticCode(),
                               opt.GetSumStatisticsMode(),
                               m_CmdLineArgs->ExecuteRemotely(),
                               db_adapter->GetFilteringAlgorithm(),
                               fmt_args->GetCustomOutputFormatSpec(),
                               m_CmdLineArgs->GetTask() == "megablast",
                               opt.GetMBIndexLoaded());
                               
        
        formatter.PrintProlog();
        
        CBatchSizeMixer mixer(2000000,   // targeted at 2M hits per batch
                              SplitQuery_GetChunkSize(opt.GetProgram())-1000);
        if (!m_CmdLineArgs->ExecuteRemotely()) {
            input.SetBatchSize(mixer.GetBatchSize());
        }

        /*** Process the input ***/
        for (; !input.End(); formatter.ResetScopeHistory()) {

            CRef<CBlastQueryVector> query_batch(input.GetNextSeqBatch(*scope));
            CRef<IQueryFactory> queries(new CObjMgr_QueryFactory(*query_batch));

            SaveSearchStrategy(args, m_CmdLineArgs, queries, opts_hndl);

            CRef<CSearchResultSet> results;

            if (m_CmdLineArgs->ExecuteRemotely()) {
                CRef<CRemoteBlast> rmt_blast = 
                    InitializeRemoteBlast(queries, db_args, opts_hndl,
                          m_CmdLineArgs->ProduceDebugRemoteOutput(),
                          m_CmdLineArgs->GetClientId());
                results = rmt_blast->GetResultSet();
            } else {
                CLocalBlast lcl_blast(queries, opts_hndl, db_adapter);
                lcl_blast.SetNumberOfThreads(m_CmdLineArgs->GetNumThreads());
                results = lcl_blast.Run();
                input.SetBatchSize(mixer.GetBatchSize(lcl_blast.GetNumExtensions()));
            }

            if (fmt_args->ArchiveFormatRequested(args)) {
                formatter.WriteArchive(*queries, *opts_hndl, *results);
            } else {
                BlastFormatter_PreFetchSequenceData(*results, scope);
                ITERATE(CSearchResultSet, result, *results) {
                    formatter.PrintOneResultSet(**result, query_batch);
                }
            }
        }

        formatter.PrintEpilog(opt);

        if (m_CmdLineArgs->ProduceDebugOutput()) {
            opts_hndl->GetOptions().DebugDumpText(NcbiCerr, "BLAST options", 1);
        }

    } CATCH_ALL(status)
    return status;
}
#endif

int CBlastnApp::Run()
{
	int status = BLAST_EXIT_SUCCESS;

	__int64 c1 = slogfile.Start();
	try {

		// Allow the fasta reader to complain on invalid sequence input
		SetDiagPostLevel(eDiag_Warning);

		/*** Get the BLAST options ***/
		const CArgs& args = GetArgs();
		CRef<CBlastOptionsHandle> opts_hndl;
		if(RecoverSearchStrategy(args, m_CmdLineArgs)){
			opts_hndl.Reset(&*m_CmdLineArgs->SetOptionsForSavedStrategy(args));
		}
		else {
			opts_hndl.Reset(&*m_CmdLineArgs->SetOptions(args));
		}

#if _MSC_VER
		output = args[kArgOutput].AsString();
		output = output.substr(0,output.find_last_of("\\")+1);
#else
		output = args[kArgOutput].AsString();
		output = output.substr(0,output.find_last_of("/")+1);
#endif
		//cout << "output" <<output << endl;

		string query_name = args[kArgQuery].AsString();
		string query_list = args[kArgQueryList].AsString();

		string logfilename;
		if (!query_list.empty())
		{
			logfilename = query_list+".log";
			args[kArgQuery].CloseFile();

		}else
		{
			logfilename = query_name +".log";
		}
		cout << logfilename <<endl;
		slogfile.m_file.open(logfilename.c_str(), fstream::out|fstream::app);
		if (!slogfile.m_file.is_open())
		{
			cout << "open log file error!"<<endl;
		}
		//////////////////////////////////////////////////////////////////////////


		const CBlastOptions& opt = opts_hndl->GetOptions();
		int initgpus = Blast_gpu_Init(opt.GetUseGpu(), opt.GetGpuID());//BlastMGPUUtil.InitGPUs(opt.GetUseGpu(), opt.GetGpuID());  
		int method = opt.GetMethod();

	///// CHANGED /////
		if (opt.GetConverted())
		{
			string dbname = args[kArgDb].AsString();
			//char* table_filename = "converted_human.1-8.wm.table";
			string table_name = dbname+".table";
			//rdsmap = new NewRecordsMap(table_name);
		}
	///// CHANGED /////

		switch(method)
		{
		case 0:
			status = Method2(opt);
			break;
		case 1:
			status = Method1(opt);
			break;
		case 2:
			status = Method3(opt);
			break;
		}

#if MULTI_QUERIES	 
		gpu_ReleaseDBMemory();
#endif
		Blast_gpu_Release();//BlastMGPUUtil.ReleaseGPUs();

		slogfile.printTotalNameBySteps();
		slogfile.m_file.close();

	} CATCH_ALL(status)
		return status;
}

int CBlastnApp::Method1_opt(string job_name)
{
	int status = BLAST_EXIT_SUCCESS;

	try {

		__int64 c1 = slogfile.Start();
		__int64 c_start = slogfile.NewStart(true);
		// Allow the fasta reader to complain on invalid sequence input
		//SetDiagPostLevel(eDiag_Warning);

		/*** Get the BLAST options ***/
		CArgs& args = GetArgs();
		CRef<CBlastOptionsHandle> opts_hndl;
		if(RecoverSearchStrategy(args, m_CmdLineArgs)){
			opts_hndl.Reset(&*m_CmdLineArgs->SetOptionsForSavedStrategy(args));
		}
		else {
			opts_hndl.Reset(&*m_CmdLineArgs->SetOptions(args));
		}
		const CBlastOptions& opt = opts_hndl->GetOptions();

		/*** Initialize the database/subject ***/
		CRef<CBlastDatabaseArgs> db_args(m_CmdLineArgs->GetBlastDatabaseArgs());
		CRef<CLocalDbAdapter> db_adapter;
		CRef<CScope> scope;
		InitializeSubject(db_args, opts_hndl, m_CmdLineArgs->ExecuteRemotely(),
			db_adapter, scope);
		_ASSERT(db_adapter && scope);

		/*** Get the query sequence(s) ***/
		CRef<CQueryOptionsArgs> query_opts = 
			m_CmdLineArgs->GetQueryOptionsArgs();
		SDataLoaderConfig dlconfig =
			InitializeQueryDataLoaderConfiguration(query_opts->QueryIsProtein(),
			db_adapter);
		CBlastInputSourceConfig iconfig(dlconfig, query_opts->GetStrand(),
			query_opts->UseLowercaseMasks(),
			query_opts->GetParseDeflines(),
			query_opts->GetRange());
		iconfig.SetQueryLocalIdMode();
		CBlastFastaInputSource fasta(m_CmdLineArgs->GetInputStream(), iconfig);
		CBlastInput input(&fasta, m_CmdLineArgs->GetQueryBatchSize());

		// Initialize the megablast database index now so we can know whether an indexed search will be run.
		// This is only important for the reference in the report, but would be done anyway.
		if (opt.GetUseIndex() && !m_CmdLineArgs->ExecuteRemotely()) {
			CRef<CBlastOptions> my_options(&(opts_hndl->SetOptions()));
			CSetupFactory::InitializeMegablastDbIndex(my_options);
		}

		/*** Get the formatting options ***/
		CRef<CFormattingArgs> fmt_args(m_CmdLineArgs->GetFormattingArgs());
		CBlastFormat formatter(opt, *db_adapter,
			fmt_args->GetFormattedOutputChoice(),
			query_opts->GetParseDeflines(),
			m_CmdLineArgs->GetInputOutputStream(),//m_CmdLineArgs->GetOutputStream(), //modified by kyzhao 
			fmt_args->GetNumDescriptions(),
			fmt_args->GetNumAlignments(),
			*scope,
			opt.GetMatrixName(),
			fmt_args->ShowGis(),
			fmt_args->DisplayHtmlOutput(),
			opt.GetQueryGeneticCode(),
			opt.GetDbGeneticCode(),
			opt.GetSumStatisticsMode(),
			m_CmdLineArgs->ExecuteRemotely(),
			db_adapter->GetFilteringAlgorithm(),
			fmt_args->GetCustomOutputFormatSpec(),
			m_CmdLineArgs->GetTask() == "megablast",
			opt.GetMBIndexLoaded());


		formatter.PrintProlog();
		CBatchSizeMixer mixer(2000000,   // targeted at 2M hits per batch
			SplitQuery_GetChunkSize(opt.GetProgram())-1000);
		if (!m_CmdLineArgs->ExecuteRemotely()) {
			input.SetBatchSize(mixer.GetBatchSize());
		}

		__int64 c2 = slogfile.End();
		slogfile.addTotalTime("Before search Prepare_time", c1, c2,false);

		/*** Process the input ***/
		//int looptimes = 0;
		for (; !input.End(); formatter.ResetScopeHistory()) {

			//printf("%d\n", looptimes++);
			c1 = slogfile.Start();

			CRef<CBlastQueryVector> query_batch(input.GetNextSeqBatch(*scope));
			CRef<IQueryFactory> queries(new CObjMgr_QueryFactory(*query_batch));

			SaveSearchStrategy(args, m_CmdLineArgs, queries, opts_hndl);

			CRef<CSearchResultSet> results;

			if (m_CmdLineArgs->ExecuteRemotely()) {
				CRef<CRemoteBlast> rmt_blast = 
					InitializeRemoteBlast(queries, db_args, opts_hndl,
					m_CmdLineArgs->ProduceDebugRemoteOutput(),
					m_CmdLineArgs->GetClientId());
				results = rmt_blast->GetResultSet();
			} else {
				CLocalBlast lcl_blast(queries, opts_hndl, db_adapter);
				size_t cpu_threads = m_CmdLineArgs->GetNumThreads();

				lcl_blast.SetNumberOfThreads( cpu_threads);
				//////////////////////////////////////////////////////////////////////////
				//pre_time
				c2 = slogfile.End();
				slogfile.addTotalTime("each prepare time",c1, c2, false);
				//////////////////////////////////////////////////////////////////////////
				c1 = slogfile.NewStart(true);
				results = lcl_blast.Run();
				c2 = slogfile.NewEnd(true);
				slogfile.addTotalTime("total search_time", c1, c2, false);
				double total_time = slogfile.elaplsedTime(c1, c2);
				cout << total_time <<"\t";
				input.SetBatchSize(mixer.GetBatchSize(lcl_blast.GetNumExtensions()));
			}

			//////////////////////////////////////////////////////////////////////////

			c1 = slogfile.Start();
			//////////////////////////////////////////////////////////////////////////

			if (fmt_args->ArchiveFormatRequested(args)) {
				formatter.WriteArchive(*queries, *opts_hndl, *results);
			} else {
				BlastFormatter_PreFetchSequenceData(*results, scope);
				ITERATE(CSearchResultSet, result, *results) {
					formatter.PrintOneResultSet(**result, query_batch);

					CConstRef<CSeq_align_set> aln_set = (**result).GetSeqAlign();
					size_t num_hits = aln_set->Get().size();
					slogfile.addTotalNum("Final hits", (long)num_hits,false);         
					//////////////////////////////////////////////////////////////////////////
				}
			}
			__int64 c2 = slogfile.End();
			//////////////////////////////////////////////////////////////////////////
			slogfile.addTotalTime("print_time",c1, c2, false);

		}
		/*if (opt.GetConverted())
		{
			string outfilename = args[kArgOutput].AsString();
			args[kArgOutput].CloseFile();
			FormatResult(outfilename);
		}*/
		__int64 t_end = slogfile.NewEnd(true);
		double total_time = slogfile.elaplsedTime(c_start, t_end);
		slogfile.addTotalTime("Total Time", total_time, false);
		slogfile.m_file<< job_name <<"\t";
		slogfile.printTotalBySteps();
		slogfile.reset();

		formatter.PrintEpilog(opt);

		if (m_CmdLineArgs->ProduceDebugOutput()) {
			opts_hndl->GetOptions().DebugDumpText(NcbiCerr, "BLAST options", 1);
		}
		//if (opt.GetConverted())
		//{
		//	string outfilename = args[kArgOutput].AsString();
		//	args[kArgOutput].CloseFile();
		//	FormatResult(outfilename);
		//}


	} CATCH_ALL(status)
		return status;
}

int CBlastnApp::Method1(const CBlastOptions& opt)
{
	int status = 0;
	string query_list_fname = opt.GetQueryList();
	if (query_list_fname.empty())
	{
		return Method1_opt("");
	}
	else
	{
		fstream query_list(query_list_fname.c_str());
		string line;
		if (query_list.is_open())
		{
			unsigned long total_jobs = 0;
			unsigned long total_works = 0;
			while (query_list.good())
			{
				getline(query_list,line);

				if (!line.empty())
				{
					total_jobs++;
					cout << line << endl;
					/*** Get the BLAST options ***/
					CArgs& args = GetArgs();
					//args[kArgOutput].CloseFile();
					args[kArgQuery].CloseFile();

					CArgDescriptions* arg_desc = m_CmdLineArgs->SetCommandLine();
					unsigned int n_plain = kMax_UInt;
					arg_desc->x_CreateArg("-query",true,line,&n_plain, args, true);
#if _MSC_VER
					string name = line.substr(line.find_last_of("\\")+1, line.length());
#else
					string name = line.substr(line.find_last_of("/")+1, line.length());
#endif
					//string outputfile = output+name+".out";
					//cout << "name:" <<outputfile <<endl;
					arg_desc->x_CreateArg("-out",true, output + name +".out",&n_plain, args, true);

					status = Method1_opt(name);
				}
			}
			query_list.close();
		}
	}
	return status;
}

int CBlastnApp::Method2_opt(string job_name)
{
	int status = BLAST_EXIT_SUCCESS; 

	try {

		__int64 c1_1,c1;
		c1 = slogfile.Start();
		c1_1 = slogfile.NewStart(true);;

		const CArgs& args = GetArgs();

		CRef<CBlastnAppArgs> lm_CmdLineArgs;
		lm_CmdLineArgs.Reset(m_CmdLineArgs);

		CRef<CBlastOptionsHandle> opts_hndl;
		if(RecoverSearchStrategy(args, lm_CmdLineArgs)){
			opts_hndl.Reset(&*lm_CmdLineArgs->SetOptionsForSavedStrategy(args));
		}
		else {
			opts_hndl.Reset(&*lm_CmdLineArgs->SetOptions(args));
		}
		const CBlastOptions& opt = opts_hndl->GetOptions();

		///*** Initialize the database/subject ***/

		CRef<CBlastDatabaseArgs> db_args(lm_CmdLineArgs->GetBlastDatabaseArgs());
		CRef<CLocalDbAdapter> db_adapter;
		CRef<CScope> scope;
		InitializeSubject(db_args, opts_hndl, lm_CmdLineArgs->ExecuteRemotely(),
			db_adapter, scope);
		_ASSERT(db_adapter && scope);

		// Initialize the megablast database index now so we can know whether an indexed search will be run.
		// This is only important for the reference in the report, but would be done anyway.
		if (opt.GetUseIndex() && !lm_CmdLineArgs->ExecuteRemotely()) {
			BlastSeqSrc* seqsrc = db_adapter->MakeSeqSrc();
			CRef<CBlastOptions> my_options(&(opts_hndl->SetOptions()));
			CSetupFactory::InitializeMegablastDbIndex(my_options);
		}
		/************************************************************************/
		/*                                                                      */
		/************************************************************************/

		/*** Get the query sequence(s) ***/
		CRef<CQueryOptionsArgs> query_opts = lm_CmdLineArgs->GetQueryOptionsArgs();
		SDataLoaderConfig dlconfig(query_opts->QueryIsProtein()); 
		dlconfig.OptimizeForWholeLargeSequenceRetrieval(); 
		CBlastInputSourceConfig iconfig(dlconfig, query_opts->GetStrand(), 
			query_opts->UseLowercaseMasks(),
			query_opts->GetParseDeflines(),
			query_opts->GetRange(),
			!lm_CmdLineArgs->ExecuteRemotely());
		iconfig.SetQueryLocalIdMode();

		CBlastFastaInputSource fasta(lm_CmdLineArgs->GetInputStream(), iconfig);
		CBlastInput input(&fasta, lm_CmdLineArgs->GetQueryBatchSize());

		/*** Get the formatting options ***/
		CRef<CFormattingArgs> fmt_args(lm_CmdLineArgs->GetFormattingArgs());
		/*** Get the formatting options ***/
		CBlastFormat formatter(opt, *db_adapter,
			fmt_args->GetFormattedOutputChoice(),
			query_opts->GetParseDeflines(),
			//lm_CmdLineArgs->GetOutputStream(),
			m_CmdLineArgs->GetInputOutputStream(),//m_CmdLineArgs->GetOutputStream(), //modified by kyzhao 
			fmt_args->GetNumDescriptions(),
			fmt_args->GetNumAlignments(),
			*scope,
			opt.GetMatrixName(),
			fmt_args->ShowGis(),
			fmt_args->DisplayHtmlOutput(),
			opt.GetQueryGeneticCode(),
			opt.GetDbGeneticCode(),
			opt.GetSumStatisticsMode(),
			lm_CmdLineArgs->ExecuteRemotely(),
			db_adapter->GetFilteringAlgorithm(),
			fmt_args->GetCustomOutputFormatSpec(),
			lm_CmdLineArgs->GetTask() == "megablast",
			opt.GetMBIndexLoaded());


		formatter.PrintProlog();
		CBatchSizeMixer mixer(2000000,   // targeted at 2M hits per batch
			SplitQuery_GetChunkSize(opt.GetProgram())-1000);
		if (!m_CmdLineArgs->ExecuteRemotely()) {
			input.SetBatchSize(mixer.GetBatchSize());
		}

		__int64 c2 = slogfile.End();
		slogfile.addTotalTime("Before search Prepare_time", c1, c2,false);

		/*** Process the input ***/
		size_t cpu_threads = lm_CmdLineArgs->GetNumThreads();
		bool is_archive = fmt_args->ArchiveFormatRequested(args);
		int k = 0;
		for (; !input.End();) {
			c1 = slogfile.Start();

			CRef<CBlastQueryVector> query_batch (input.GetNextSeqBatch(*scope));
			CRef<IQueryFactory> queries(new CObjMgr_QueryFactory(*query_batch));

			SaveSearchStrategy(args, lm_CmdLineArgs, queries, opts_hndl);			
			CLocalBlast* lcl_blast = new CLocalBlast(queries, opts_hndl, db_adapter);

			//////////////////////////////////////////////////////////////////////////
			//pre_time
			c2 = slogfile.End();
			slogfile.addTotalTime("each prepare time",c1, c2, false);

			work_item * item = new work_item();
			item->query_batch = query_batch;
			item->queries = queries;
			item->m_lbast = lcl_blast;
			//item->initgpus = initgpus;
			//item->is_gpuonly = isgpuonly;
			item->cpu_threads = cpu_threads;

			item->formatter = &formatter;
			item->is_archive = is_archive;
			item->scope = scope;
			item->opts_hndl = opts_hndl;

			query_queue->add(item);
			k++;
		}

		PrintThread* p_print_thread = (PrintThread*)&print_thread[0];
		while (p_print_thread->GetProcessNum() != k)
		{
			//cout << ".";
#ifdef __linux__
			usleep(3);
#endif
		}
		//					cout << endl;
		__int64 c2_1 = slogfile.NewEnd(true);
		double total_time = slogfile.elaplsedTime(c1_1, c2_1);
		slogfile.addTotalTime("Total Time", total_time, false);
		slogfile.m_file<< job_name <<"\t";
		slogfile.printTotalBySteps();
		slogfile.reset();

		p_print_thread->ResetProcessNum();

		formatter.PrintEpilog(opt);								  
		if (lm_CmdLineArgs->ProduceDebugOutput()) {
			opts_hndl->GetOptions().DebugDumpText(NcbiCerr, "BLAST options", 1);
		}

		//if (opt.GetConverted())
		//{
		//	string outfilename = args[kArgOutput].AsString();
		//	args[kArgOutput].CloseFile();
		//	FormatResult(outfilename);
		//}

	} CATCH_ALL(status)
		return status;
}
// version before 2013.8.7
int CBlastnApp::Method2(const CBlastOptions& opt)
{
	int status = BLAST_EXIT_SUCCESS;

	try {

		int prelim_num = opt.GetPrelimNum();
		int trace_num = opt.GetTraceNum();
		int print_num = opt.GetPrintNum();

		//////////////////////////////////////////////////////////////////////////
		query_queue = new work_queue<work_item*>;
		prelim_queue = new work_queue<work_item*>;
		result_queue = new work_queue<work_item*>;


		prelim_thread = new PrelimSearchThread[prelim_num];
		trace_thread = new TraceBackThread[trace_num];
		print_thread = new PrintThread[print_num];


		for (int i = 0; i < prelim_num; i++)
		{
			prelim_thread[i].InitQueue(query_queue, prelim_queue);
			prelim_thread[i].start();
		}

		for (int i = 0; i < trace_num; i++)
		{
			trace_thread[i].InitQueue(prelim_queue, result_queue);
			trace_thread[i].start();
		}

		for (int i = 0; i < print_num; i++)
		{
			PrintThread* p = (PrintThread*)&print_thread[i];
			p->InitQueue(result_queue);
			print_thread[i].start();
		}

		string query_list_fname = opt.GetQueryList();
		if (query_list_fname.empty())
		{
			return Method2_opt("");
		}
		else
		{
			fstream query_list(query_list_fname.c_str());
			string line;
			if (query_list.is_open())
			{
				unsigned long total_jobs = 0;
				while (query_list.good())
				{
					getline(query_list,line);

					if (!line.empty())
					{
						total_jobs++;
						cout << line << endl;
						/*** Get the BLAST options ***/
						CArgs& args = GetArgs();

						//args[kArgOutput].CloseFile();
						args[kArgQuery].CloseFile();

						CArgDescriptions* arg_desc = m_CmdLineArgs->SetCommandLine();
						unsigned int n_plain = kMax_UInt;
						arg_desc->x_CreateArg("-query",true,line,&n_plain, args, true);

#if _MSC_VER
						string name = line.substr(line.find_last_of("\\")+1, line.length());
#else
						string name = line.substr(line.find_last_of("/")+1, line.length());
#endif
						//cout << "name:" << name <<endl;;
						arg_desc->x_CreateArg("-out",true, output + name +".out",&n_plain, args, true);

						Method2_opt(name);
					}
				}
				query_list.close();
			}
		}

		for (int i = 0; i < prelim_num; i++)
		{
			prelim_thread[i].stop();
			//prelim_thread[i].join();
		}

		for (int i = 0; i < trace_num; i++)
		{
			trace_thread[i].stop();
			//trace_thread[i].join();
		}
		for (int i = 0; i < print_num; i++)
		{
			print_thread[i].stop();
			//print_thread[i].join();
		}
	} CATCH_ALL(status)
		return status;
}

//query pipline version
int CBlastnApp::Method3_opt(string job_name)
{
	int status = BLAST_EXIT_SUCCESS;

	try
	{
		JobItem* job_item = new JobItem();
		job_item->current_work_num = 0;

		__int64 c1 = slogfile.NewStart(true);
		job_item->start_time = c1;

		CArgs& args = GetArgs();
		CArgs * p_args = new CArgs(args);
		job_item->p_args = p_args;
		CRef<CBlastnAppArgs> lm_CmdLineArgs;
		lm_CmdLineArgs.Reset(m_CmdLineArgs);

		CRef<CBlastOptionsHandle> opts_hndl;
		if(RecoverSearchStrategy(args, lm_CmdLineArgs)){
			opts_hndl.Reset(&*lm_CmdLineArgs->SetOptionsForSavedStrategy(args));
		}
		else {
			opts_hndl.Reset(&*lm_CmdLineArgs->SetOptions(args));
		}
		const CBlastOptions& opt = opts_hndl->GetOptions();

		///*** Initialize the database/subject ***/
		CRef<CBlastDatabaseArgs> db_args(lm_CmdLineArgs->GetBlastDatabaseArgs());
		CRef<CLocalDbAdapter> db_adapter;
		CRef<CScope> scope;
		InitializeSubject(db_args, opts_hndl, lm_CmdLineArgs->ExecuteRemotely(),
			db_adapter, scope);
		_ASSERT(db_adapter && scope);

		// Initialize the megablast database index now so we can know whether an indexed search will be run.
		// This is only important for the reference in the report, but would be done anyway.
		if (opt.GetUseIndex() && !lm_CmdLineArgs->ExecuteRemotely()) {
			BlastSeqSrc* seqsrc = db_adapter->MakeSeqSrc();
			CRef<CBlastOptions> my_options(&(opts_hndl->SetOptions()));
			CSetupFactory::InitializeMegablastDbIndex(my_options);
		}
		/************************************************************************/
		/*                                                                      */
		/************************************************************************/

		/*** Get the query sequence(s) ***/
		CRef<CQueryOptionsArgs> query_opts = lm_CmdLineArgs->GetQueryOptionsArgs();
		SDataLoaderConfig dlconfig(query_opts->QueryIsProtein()); 
		dlconfig.OptimizeForWholeLargeSequenceRetrieval(); 
		CBlastInputSourceConfig iconfig(dlconfig, query_opts->GetStrand(), 
			query_opts->UseLowercaseMasks(),
			query_opts->GetParseDeflines(),
			query_opts->GetRange(),
			!lm_CmdLineArgs->ExecuteRemotely());
		iconfig.SetQueryLocalIdMode();

		CBlastFastaInputSource fasta(lm_CmdLineArgs->GetInputStream(), iconfig);
		CBlastInput input(&fasta, lm_CmdLineArgs->GetQueryBatchSize());

		/*** Get the formatting options ***/
		CRef<CFormattingArgs> fmt_args(lm_CmdLineArgs->GetFormattingArgs());
		/*** Get the formatting options ***/
		CBlastFormat* formatter = new CBlastFormat(opt, *db_adapter,
			fmt_args->GetFormattedOutputChoice(),
			query_opts->GetParseDeflines(),
			m_CmdLineArgs->GetInputOutputStream(),//m_CmdLineArgs->GetOutputStream(), //modified by kyzhao 
			fmt_args->GetNumDescriptions(),
			fmt_args->GetNumAlignments(),
			*scope,
			opt.GetMatrixName(),
			fmt_args->ShowGis(),
			fmt_args->DisplayHtmlOutput(),
			opt.GetQueryGeneticCode(),
			opt.GetDbGeneticCode(),
			opt.GetSumStatisticsMode(),
			lm_CmdLineArgs->ExecuteRemotely(),
			db_adapter->GetFilteringAlgorithm(),
			fmt_args->GetCustomOutputFormatSpec(),
			lm_CmdLineArgs->GetTask() == "megablast",
			opt.GetMBIndexLoaded());

		//cout << "formatter 1:" << formatter <<" out "<< lm_CmdLineArgs->GetOutputStream() << endl;
		formatter->PrintProlog();
		CBatchSizeMixer mixer(2000000,   // targeted at 2M hits per batch
			SplitQuery_GetChunkSize(opt.GetProgram())-1000);
		if (!m_CmdLineArgs->ExecuteRemotely()) {
			input.SetBatchSize(mixer.GetBatchSize());
		}
		//__int64 c2 = slogfile.End();
		//slogfile.addTotalTime("Before search Prepare_time", c1, c2,false);

		/*** Process the input ***/
		size_t cpu_threads = lm_CmdLineArgs->GetNumThreads();
		bool is_archive = fmt_args->ArchiveFormatRequested(args);
		int k = 0;
		//int looptimes = 0;	

		for (; !input.End(); /*formatter.ResetScopeHistory()*/) {
			//	printf("%d\n", looptimes++);
			//c1 = slogfile.Start();

			CRef<CBlastQueryVector> query_batch (input.GetNextSeqBatch(*scope));
			CRef<IQueryFactory> queries(new CObjMgr_QueryFactory(*query_batch));

			SaveSearchStrategy(args, lm_CmdLineArgs, queries, opts_hndl);			
			CLocalBlast* lcl_blast = new CLocalBlast(queries, opts_hndl, db_adapter);

			//////////////////////////////////////////////////////////////////////////
			//pre_time
			//c2 = slogfile.End();
			//slogfile.addTotalTime("each prepare time",c1, c2, false);

			work_item * item = new work_item();
			item->query_batch = query_batch;
			item->queries = queries;
			item->m_lbast = lcl_blast;
			//item->initgpus = initgpus;

			item->cpu_threads = cpu_threads;
			item->formatter = formatter;
			item->is_archive = is_archive;
			item->scope = scope;
			item->opts_hndl = opts_hndl;

			item->p_job_item = job_item;

			query_queue->add(item);
			k++;
		}
		//////////////////////////////////////////////////////////////////////////
		//put job into job queue
		job_item->work_num = k;
		job_item->db_args= db_args;
		job_item->jobname = job_name;
		job_item->db_adapter = db_adapter;
		job_item->fmt_args = fmt_args;
		job_item->lm_CmdLineArgs = lm_CmdLineArgs;
		job_item->opts_hndl = opts_hndl;
		job_item->query_opts = query_opts;
		job_item->scope =scope;
		job_item->formatter = formatter;

		job_queue->add(job_item);

	}CATCH_ALL(status)
		return status; 
}
int CBlastnApp::Method3(const CBlastOptions& opt)
{
	int status = BLAST_EXIT_SUCCESS;


	try {
		__int64 c1_1 = slogfile.NewStart(true);

		int prelim_num = opt.GetPrelimNum();
		int trace_num = opt.GetTraceNum();
		int print_num = opt.GetPrintNum();

		//cout << "prelim:" << prelim_num << " trace:" << trace_num << "print:" << print_num << endl;
		//////////////////////////////////////////////////////////////////////////
		job_queue = new work_queue<JobItem*>;
		query_queue = new work_queue<work_item*>;
		prelim_queue = new work_queue<work_item*>;
		result_queue = new work_queue<work_item*>;

		prelim_thread = new PrelimSearchThread[prelim_num];
		trace_thread = new TraceBackThread[trace_num];
		print_thread = new PrintThread_1[print_num];


		for (int i = 0; i < prelim_num; i++)
		{
			prelim_thread[i].InitQueue(query_queue, prelim_queue);
			prelim_thread[i].start();
		}

		for (int i = 0; i < trace_num; i++)
		{
			trace_thread[i].InitQueue(prelim_queue, result_queue);
			trace_thread[i].start();
		}

		for (int i = 0; i < print_num; i++)
		{
			PrintThread_1* p = (PrintThread_1*)&print_thread[i];
			p->InitQueue(result_queue);
			p->SetRecord(rdsmap);
			p->start();
		}

		string query_list_fname = opt.GetQueryList();
		if (query_list_fname.empty())
		{
			return Method3_opt("");
		}
		else
		{
			fstream query_list(query_list_fname.c_str());
			string line;
			if (query_list.is_open())
			{
				unsigned long total_jobs = 0;
				while (query_list.good())
				{
					getline(query_list,line);

					if (!line.empty())
					{
						total_jobs++;
						//cout << line << endl;
						/*** Get the BLAST options ***/
						CArgs& args = GetArgs();

						//args[kArgOutput].CloseFile();
						//args[kArgQuery].CloseFile();

						CArgDescriptions* arg_desc = m_CmdLineArgs->SetCommandLine();
						unsigned int n_plain = kMax_UInt;
						arg_desc->x_CreateArg("-query",true,line,&n_plain, args, true);
#if _MSC_VER
						string name = line.substr(line.find_last_of("\\")+1, line.length());
#else
						string name = line.substr(line.find_last_of("/")+1, line.length());
#endif
						arg_desc->x_CreateArg("-out",true, output + name +".out",&n_plain, args, true);

						Method3_opt(name);
					}
				}
				query_list.close();

				PrintThread_1* p = (PrintThread_1*)&print_thread[0];

				while (p->GetTotalJobNum() != total_jobs)
				{
#ifdef __linux__
					usleep(3);
#endif
				}
			}
		}

		for (int i = 0; i < prelim_num; i++)
		{
			prelim_thread[i].stop();
			//prelim_thread[i].join();
		}

		for (int i = 0; i < trace_num; i++)
		{
			trace_thread[i].stop();
			//trace_thread[i].join();
		}
		for (int i = 0; i < print_num; i++)
		{
			print_thread[i].stop();
			//print_thread[i].join();
		}

		__int64 t_end = slogfile.NewEnd(true);
		//slogfile.m_file << "Total" << "\t" << slogfile.elaplsedTime(c1_1, t_end) <<endl;
		double total_time = slogfile.elaplsedTime(c1_1, t_end);
		//slogfile.addTotalTime("Total Time", total_time, false);
		slogfile.m_file<< "Total" <<"\t" << total_time << "\n";

	} CATCH_ALL(status)
		return status;
}

#ifndef SKIP_DOXYGEN_PROCESSING
int main(int argc, const char* argv[] /*, const char* envp[]*/)
{
#if (!OPT_TRACEBACK && LOGFILE)
	string logfilename = argv[2];
	logfilename = "cpu_"+logfilename +".log.txt";

	slogfile.m_file.open(logfilename.c_str(), fstream::out|fstream::app);

	__int64 c1 = slogfile.Start();
	
	int ret = CBlastnApp().AppMain(argc, argv, 0, eDS_Default, 0);

	//*************************** added by kyzhao  for gpu log
	__int64 c2 = slogfile.End();
	slogfile.addTotalTime("Total Time", c1, c2, false);
	//*************************** added by kyzhao for gpu logo
	return ret;
	//slogfile.m_file << "Total" << ": ";
#else
	return CBlastnApp().AppMain(argc, argv, 0, eDS_Default, 0);
#endif
}
#endif /* SKIP_DOXYGEN_PROCESSING */
