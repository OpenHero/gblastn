#include <algo/blast/gpu_blast/work_thread.hpp>
#include <app/blast/blast_app_util.hpp>
#include <algo/blast/api/objmgr_query_data.hpp>
#include <algo/blast/format/blast_format.hpp>

#include <iostream>
#include <fstream>
using namespace std;

BEGIN_NCBI_SCOPE
BEGIN_SCOPE(blast)
USING_SCOPE(objects);

//////////////////////////////////////////////////////////////////////////

RETURN_INT AlignmentThread::run()
{
	cout << "Alignment thread is running..." << endl;
	while(m_running) {

		work_item* item = query_queue->remove();
		CRef<CSearchResultSet> results;

		CLocalBlast* lcl_blast = item->m_lbast;	

		is_gpuonly = item->is_gpuonly;
		initgpus = item->initgpus;
		cpu_threads = item->cpu_threads;

		int total_threads = cpu_threads;

		if (is_gpuonly && initgpus > 0)
		{
			total_threads = initgpus;
		}else
		{
			total_threads += initgpus;
		}

		lcl_blast->SetNumberOfThreads( total_threads);

		//////////////////////////////////////////////////////////////////////////
		__int64 c1 = slogfile.Start();
		//results = lcl_blast->Run();
		if (lcl_blast->PrelimSearchRun(results) == 0)
		{
			results = lcl_blast->TraceBackRun();
		}
		__int64 c2 = slogfile.End();
		slogfile.addTotalTime("total search_time", c1, c2, false);

		cout << "Alignment one request...." <<endl;

		item->results = results;			
		result_queue->add(item);
	}
	return NULL;
}

RETURN_INT PrelimSearchThread::run()
{
	cout << "PrelimSearch thread is running..." << endl;
	while(m_running) {
		
		work_item* item = query_queue->remove();
		CRef<CSearchResultSet> results;

		CLocalBlast* lcl_blast = item->m_lbast;	

		cpu_threads = item->cpu_threads;

		//cout << "prelim query :" << item->p_job_item->jobname << endl;

		lcl_blast->SetNumberOfThreads( cpu_threads);

		item->status = lcl_blast->PrelimSearchRun(results);

		//cout << "PrelimSearched one request...." <<endl;

		item->results = results;			
		prelim_queue->add(item);
	}
	return NULL;
}

RETURN_INT TraceBackThread::run()
{
	cout << "TraceBack thread is running..." << endl;
	while(m_running) {

		work_item* item = prelim_queue->remove();
		
		//cout << "TraceBacking " << item->p_job_item->jobname <<endl;

		CLocalBlast* lcl_blast = item->m_lbast;	

		int status = item->status;
		if (0 == status)
		{
		   item->results = lcl_blast->TraceBackRun();
		}

		//cout << "TraceBacked one request...." <<endl;
		result_queue->add(item);
	}
	return NULL;
}

RETURN_INT PrintThread::run()
{
	cout << "Print thread is running..." << endl;
	while(m_running)
	{
		work_item* item = result_queue->remove();
		CRef<CBlastQueryVector> query_batch = item->query_batch;
		CRef<IQueryFactory> queries = item->queries;
		CRef<CSearchResultSet> results = item->results;

		formatter = item->formatter;
		opts_hndl = item->opts_hndl;
		scope = item->scope;
		is_archive = item->is_archive;

		cout << "query_batch" << query_batch << endl;
		cout << "queries" << queries << endl;
		//////////////////////////////////////////////////////////////////////////
		__int64 c1 = slogfile.Start();
		if (is_archive){
			formatter->WriteArchive(*queries, *opts_hndl, *results);
		} else {
			BlastFormatter_PreFetchSequenceData(*results, scope);
			ITERATE(CSearchResultSet, result, *results) {
				formatter->PrintOneResultSet(**result, query_batch);

				CConstRef<CSeq_align_set> aln_set = (**result).GetSeqAlign();
				int num_hits = aln_set->Get().size();
				slogfile.addTotalNum("Final hits", num_hits,false);       
				//////////////////////////////////////////////////////////////////////////
			}
		}
		__int64 c2 = slogfile.End();
		slogfile.addTotalTime("print_time",c1, c2, false);

		delete item->m_lbast;	
		delete item;

		
		process_num++;

		m_total_works++;

		//cout << "Print one request...." <<endl;
		
	}
	return NULL;
}

void PrintThread_1::SetRecord(NewRecordsMap* in_rdsmap)
{
	rdsmap = in_rdsmap;
}
int PrintThread_1::FormatResult(string outfile)
{
	if (rdsmap == NULL)
	{
		printf("Can't load record map!\n");
		return -1;
		//exit(-1);
	}
	///// CHANGED /////
	//string outputfile = args[kArgOutput].AsString();
	//printf("Output file : %s\n", outputfile.c_str());
	//printf("Output Choice : %d\n", m_CmdLineArgs->GetFormattingArgs()->GetFormattedOutputChoice());
	// Flush all content first
	//cout << "outfile" <<outfile<<endl;
	fstream ioStream;
	ioStream.open(outfile.c_str());

	// Return to the beginning
	//ioStream.seekg(0);

	std::vector<std::string> lines;
	std::string line;
	int line_count = 0;
	while (std::getline(ioStream, line))
	{
		lines.push_back(line);
	}
	printf("Line Count : %d\n", lines.size());

	// Return to the beginning
	ioStream.clear();
	ioStream.seekg(0);

	for ( int i = 0; i < lines.size(); i++ ) {
		if ( lines[i][0] == '#' ) {
			ioStream.write(lines[i].c_str(), lines[i].length());
			ioStream.write("\n", 1);
			//fprintf(ioStream, "%s", line);
		}
		else {
			std::vector<std::string> tokens;
			char* pch = strtok((char*)lines[i].c_str(),"\t");

			int k = 0;
			while ( pch != NULL ) {
				tokens.push_back(std::string(pch));
				pch = strtok (NULL, "\t");
				k++;
			}

			//cout << "k:"<< k<<endl;
			unsigned int start_offset = atoi(tokens[8].c_str());
			unsigned int end_offset = atoi(tokens[9].c_str());

			NewRecord* found = rdsmap->getCorrectedRecord(tokens[1], start_offset, end_offset);
			if ( found != NULL ) {
				std::string buf;
				char start_offset_s[50];
				char end_offset_s[50];
				sprintf(start_offset_s, "%d", start_offset);
				sprintf(end_offset_s, "%d", end_offset);

				buf.append(tokens[0]).append("\t")
					.append(found->_id).append("\t")
					.append(tokens[2]).append("\t")
					.append(tokens[3]).append("\t")
					.append(tokens[4]).append("\t")
					.append(tokens[5]).append("\t")
					.append(tokens[6]).append("\t")
					.append(tokens[7]).append("\t")
					.append(start_offset_s).append("\t")
					.append(end_offset_s).append("\t")
					.append(tokens[10]).append("\t")
					.append(tokens[11]).append("\n");
				ioStream.write(buf.c_str(), buf.length());
				//fprintf(fo, "%s", buf.c_str());
			}
			else
				continue;
		}
	}

	ioStream.close();
	printf("Complete Correcting!\n");
	///// CHANGED /////
	return 0;
}

RETURN_INT PrintThread_1::run()
{
	cout << "Print thread is running..." << endl;
	while(m_running)
	{
		work_item* item = result_queue->remove();
		CRef<CBlastQueryVector> query_batch = item->query_batch;
		CRef<IQueryFactory> queries = item->queries;
		CRef<CSearchResultSet> results = item->results;

		formatter = item->formatter;
		opts_hndl = item->opts_hndl;
		scope = item->scope;
		is_archive = item->is_archive;

		//cout << "formatter 2:" << formatter << endl;
		//cout << "queries" << queries << endl;
		//////////////////////////////////////////////////////////////////////////
		if (is_archive){
			formatter->WriteArchive(*queries, *opts_hndl, *results);
		} else {
			BlastFormatter_PreFetchSequenceData(*results, scope);
			ITERATE(CSearchResultSet, result, *results) {
				formatter->PrintOneResultSet(**result, query_batch);
				//////////////////////////////////////////////////////////////////////////
			}
		}

		JobItem* job_item = item->p_job_item;
		job_item->current_work_num++;
		
		//cout << job_item->jobname << ":" << m_total_jobs << endl;

		if (job_item->current_work_num == job_item->work_num)
		{
			const CBlastOptions& opt = opts_hndl->GetOptions();

			CRef<CBlastnAppArgs> lm_CmdLineArgs = job_item->lm_CmdLineArgs;

			formatter->PrintEpilog(opt);								  
			if (lm_CmdLineArgs->ProduceDebugOutput()) {
				opts_hndl->GetOptions().DebugDumpText(NcbiCerr, "BLAST options", 1);
			}
			
			//FormatResult(lm_CmdLineArgs);
			if (opt.GetConverted())
			{
				CArgs& args = *job_item->p_args;
				if (lm_CmdLineArgs->GetFormattingArgs()->GetFormattedOutputChoice() == 7)
				{
					string outfile = args[kArgQuery].AsString();
					args[kArgQuery].CloseFile();
					FormatResult(outfile);
				}
			}

			delete formatter;
			delete job_item->p_args;

			m_total_jobs++;
			//cout << job_item->jobname << ":" << m_total_jobs << endl;
			delete job_item;
		}

		delete item->m_lbast;	
		delete item;

		//cout << "Print one request...." <<endl;

	}
	return NULL;
}

//////////////////////////////////////////////////////////////////////////

RETURN_INT PrepareThread::run()
{
	cout << "Prepare thread is running..." << endl;
	while(m_running)
	{
		JobItem* job_item = job_queue->remove();

		CArgs& args = *(job_item->p_args);

		CRef<CBlastnAppArgs> lm_CmdLineArgs = job_item->lm_CmdLineArgs;
		CRef<CBlastOptionsHandle> opts_hndl = job_item->opts_hndl;
		
		const CBlastOptions& opt = opts_hndl->GetOptions();

		CRef<CBlastDatabaseArgs> db_args = job_item->db_args;
		CRef<CLocalDbAdapter> db_adapter = job_item->db_adapter;
		CRef<CScope> scope = job_item->scope;
		

		CRef<CQueryOptionsArgs> query_opts = job_item->query_opts;

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
			lm_CmdLineArgs->GetOutputStream(),
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


		formatter->PrintProlog();

		job_item->fmt_args = fmt_args;
		job_item->formatter = formatter;

		//////////////////////////////////////////////////////////////////////////
		//GPU
		int initgpus = job_item->initgpus;

		/*** Process the input ***/
		//bool isgpuonly  = opt.GetUseGpuOnly();
		int cpu_threads = lm_CmdLineArgs->GetNumThreads();
		bool is_archive = fmt_args->ArchiveFormatRequested(args);
		int k = 0;
		//int looptimes = 0;	

		for (; !input.End(); /*formatter.ResetScopeHistory()*/) {
			//	printf("%d\n", looptimes++);
			//__int64 c1 = slogfile.Start();

			CRef<CBlastQueryVector> query_batch (input.GetNextSeqBatch(*scope));
			CRef<IQueryFactory> queries(new CObjMgr_QueryFactory(*query_batch));

			SaveSearchStrategy(args, lm_CmdLineArgs, queries, opts_hndl);			
			CLocalBlast* lcl_blast = new CLocalBlast(queries, opts_hndl, db_adapter);

			//////////////////////////////////////////////////////////////////////////
			//pre_time
			//__int64 c2 = slogfile.End();
			//slogfile.addTotalTime("each prepare time",c1, c2, false);

			work_item * item = new work_item();
			item->query_batch = query_batch;
			item->queries = queries;
			item->m_lbast = lcl_blast;
			item->initgpus = initgpus;
			//item->is_gpuonly = isgpuonly;
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

		cout << "Prepared one request:" << job_item->jobname <<":"<< k <<endl;

	}
	return NULL;
}

END_SCOPE(blast)
END_NCBI_SCOPE
