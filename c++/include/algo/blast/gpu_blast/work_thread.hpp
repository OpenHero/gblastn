#ifndef __WORK_THREAD_H__
#define __WORK_THREAD_H__
#include <objmgr/object_manager.hpp>
//#include <objtools/blast/seqdb_reader/seqdb.hpp>
#include <algo/blast/blastinput/blast_args.hpp>
#include <algo/blast/blastinput/blastn_args.hpp>
#include <algo/blast/api/local_blast.hpp>
#include <algo/blast/format/blast_format.hpp>
#include <algo/blast/blastinput/blast_fasta_input.hpp>


#include <algo/blast/gpu_blast/work_thread_base.hpp>
#include <algo/blast/gpu_blast/thread_work_queue.hpp>

#include <algo/blast/gpu_blast/gpu_logfile.h>
#include <algo/blast/gpu_blast/utility.h>

BEGIN_NCBI_SCOPE
BEGIN_SCOPE(blast)

//////////////////////////////////////////////////////////////////////////

typedef struct JobItem
{
	string		jobname; 
	int			work_num; // account the work items number;
	int			current_work_num; // current work items number;
	int			initgpus;

//#if LOG_TIME
	__int64		start_time;
	__int64		end_time;
//#endif

	//////////////////////////////////////////////////////////////////////////
	CRef<CBlastnAppArgs> lm_CmdLineArgs;

	CRef<CBlastOptionsHandle> opts_hndl;
	//CBlastOptions opt;

	CRef<CBlastDatabaseArgs> db_args;
	CRef<CLocalDbAdapter> db_adapter;
	CRef<CScope> scope;
	/*** Get the query sequence(s) ***/
	CRef<CQueryOptionsArgs> query_opts;
	/*** Get the formatting options ***/
	CRef<CFormattingArgs> fmt_args;
	/*** Get the formatting options ***/
	CBlastFormat* formatter;
	CArgs* p_args;


}JobItem;

//////////////////////////////////////////////////////////////////////////
typedef struct work_item
{
	//////////////////////////////////////////////////////////////////////////
	JobItem*					p_job_item;
	//////////////////////////////////////////////////////////////////////////
	CRef<CBlastQueryVector> query_batch;
	CRef<IQueryFactory> queries;
	CLocalBlast*		m_lbast;
	CRef<CSearchResultSet> results;
	int							initgpus;
	bool						is_gpuonly;
	int							cpu_threads;

	int							status;

	//////////////////////////////////////////////////////////////////////////
	CBlastFormat*				formatter;
	bool						is_archive;
	CRef<CScope>				scope;
	CRef<CBlastOptionsHandle>	opts_hndl;

}work_item;


class AlignmentThread : public WorkThreadBase
{
	work_queue<work_item*>		*query_queue;
	work_queue<work_item*>		*result_queue;
	int							initgpus;
	bool						is_gpuonly;
	int							cpu_threads;
public:

	AlignmentThread(){}
	
	void InitQueue(work_queue<work_item* >* qy_queue, 
		work_queue<work_item*>* ret_queue)
	{
		query_queue = qy_queue;
		result_queue = ret_queue;
	}

	RETURN_INT run();

};

class PrelimSearchThread : public WorkThreadBase
{
	work_queue<work_item*>		*query_queue;
	work_queue<work_item*>		*prelim_queue;
	int							initgpus;
	bool						is_gpuonly;
	int							cpu_threads;
public:

	PrelimSearchThread(){}
	void InitQueue(work_queue<work_item* >* qy_queue, 
		work_queue<work_item*>* pre_queue)
	{
		query_queue = qy_queue; 
		prelim_queue = pre_queue;
	}

	RETURN_INT run();

};

class TraceBackThread : public WorkThreadBase
{
	work_queue<work_item*>		*prelim_queue;
	work_queue<work_item*>		*result_queue;
	int							initgpus;
	bool						is_gpuonly;
	int							cpu_threads;
public:

	TraceBackThread(){}

	void InitQueue(work_queue<work_item* >* pre_queue,	work_queue<work_item*>* ret_queue) 
	{
		prelim_queue = pre_queue;
		result_queue = ret_queue;
	}


	RETURN_INT run();

};

class PrintThread : public WorkThreadBase
{
	work_queue<work_item*>		*result_queue;
	CBlastFormat*				formatter;
	bool						is_archive;
	CRef<CScope>				scope;
	CRef<CBlastOptionsHandle>	opts_hndl;
	int							process_num;
	unsigned long				m_total_works;
public:

	PrintThread()
		: process_num(0),
		m_total_works(0){}
	void InitQueue(work_queue<work_item*>*	ret_queue) 
	{
		result_queue = ret_queue;
	}

	void ResetProcessNum(){process_num = 0;}
	int GetProcessNum(){return process_num;}

	unsigned long GetTotalWorkNum(){return m_total_works;}


	RETURN_INT run();
};

class PrintThread_1 : public WorkThreadBase
{
	work_queue<work_item*>		*result_queue;
	CBlastFormat*				formatter;
	bool						is_archive;
	CRef<CScope>				scope;
	CRef<CBlastOptionsHandle>	opts_hndl;
	int							process_num;
	unsigned long				m_total_jobs;

	NewRecordsMap* rdsmap;
public:

	PrintThread_1() 
		: m_total_jobs(0){}
	void InitQueue(work_queue<work_item*>*	ret_queue)
	{
		result_queue = ret_queue;
	}

	unsigned long GetTotalJobNum(){return m_total_jobs;}

	void SetRecord(NewRecordsMap* int_rdsmap);
	int FormatResult(string outfile);


	RETURN_INT run();
};

//////////////////////////////////////////////////////////////////////////
// process each query command
class PrepareThread : public WorkThreadBase
{
	work_queue<JobItem*>		*job_queue;
	work_queue<work_item*>		*query_queue;

public:

	PrepareThread(){}
	void InitQueue(work_queue<JobItem* >* jb_queue,
		work_queue<work_item*>		*qy_queue)
	{
		job_queue = jb_queue;
		query_queue = qy_queue;
	}

	RETURN_INT run();

};

class FinishJobThread : public WorkThreadBase
{
	work_queue<JobItem*>*		job_queue;
public:
	FinishJobThread(){}
	void InitQueue(work_queue<JobItem*>* jb_queue)
	{
		job_queue = jb_queue;
	}
	RETURN_INT run();
};

//////////////////////////////////////////////////////////////////////////
//thread pool

END_SCOPE(blast)
END_NCBI_SCOPE

#endif // __WORK_THREAD_H__
