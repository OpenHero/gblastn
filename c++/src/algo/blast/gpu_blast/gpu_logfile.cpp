// Utilities and timing functions
//#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <algo/blast/gpu_blast/gpu_logfile.h>
#include <cuda.h>
#include <cuda_runtime.h>
//#include <helper_cuda.h>

#ifndef _MSC_VER
time_type getSystemTime() {

	struct timeval  start_time;
	gettimeofday(&start_time, 0);
	return (1000000 * (start_time.tv_sec) + start_time.tv_usec);	 //ms
}
#endif

static cudaEvent_t start, stop;

CLogFile::CLogFile()
{
#ifdef _MSC_VER
	QueryPerformanceFrequency(&large_interger); 
	dff = (double)large_interger.QuadPart; 
#endif

#if TIME_GPU
	sdkCreateTimer(&timer);
#endif

}

CLogFile::~CLogFile()
{
	if (m_file.is_open())
	{
		m_file.close();
	}

#if TIME_GPU
	sdkDeleteTimer(&timer);
#endif
}

time_type CLogFile::NewStart(bool is_log)
{ 
	if (is_log)
	{
#ifdef _MSC_VER
		QueryPerformanceCounter(&large_interger);  
		c1 = large_interger.QuadPart;
#else
		c1 = getSystemTime();
#endif
	}
	return c1;
}

time_type CLogFile::NewEnd(bool is_log)
{
	if (is_log)
	{
#ifdef _MSC_VER
		QueryPerformanceCounter(&large_interger);  
		c2 = large_interger.QuadPart;
#else
		c2 = getSystemTime();
#endif
	}
	return c2;
}

time_type CLogFile::Start()
{
#if LOG_TIME
#ifdef _MSC_VER
	QueryPerformanceCounter(&large_interger);  
	c1 = large_interger.QuadPart;
#else
	c1 = getSystemTime();
#endif

#endif
	return c1;
}

time_type CLogFile::End()
{
#if LOG_TIME
#ifdef _MSC_VER
	QueryPerformanceCounter(&large_interger);  
	c2 = large_interger.QuadPart;
#else
	c2 = getSystemTime();
#endif

#endif 
	return c2;
}

double CLogFile::printElaplsedTime(string id, time_type in_c1, time_type in_c2)
{
	double ela_time = elaplsedTime(in_c1, in_c2);
	m_file << id << "\t" << ela_time<<"\n";
	return ela_time;
}

double CLogFile::elaplsedTime(time_type in_c1, time_type in_c2)
{
#ifdef _MSC_VER					  
	return (double)(in_c2 - in_c1) * 1000 / dff;
#else 
	return (double)(in_c2- in_c1)/1000;
#endif
}

double CLogFile::elaplsedTime()
{
#ifdef _MSC_VER
	return (double)(c2 - c1) * 1000 / dff;
#else
	return (double)(c2-c1)/1000;
#endif
}

void CLogFile::KernelStart()
{
#if TIME_GPU
	sdkResetTimer(&timer);
   sdkStartTimer(&timer);
#endif
}

void CLogFile::KernelEnd()
{
#if TIME_GPU
	checkCudaErrors(cudaDeviceSynchronize());
	sdkStopTimer(&timer);
#endif
}

double CLogFile::KernelElaplsedTime()
{
#if TIME_GPU
	m_eptime = sdkGetTimerValue(&timer);
	//m_file<< kernelName <<"\t" << m_eptime<<endl;
	return (double)m_eptime;
#endif
	return 0;
}

void CLogFile::addTotalTime(string id, double time, bool is_print)
{

	if (is_print)
	{
		cout << id << "\t" << time <<"\t"; 
	}	
#if LOG_TIME	
	timemap::iterator itr = total_time_map.find(id);
	
	if (itr == total_time_map.end())
	{
		total_time_map.insert(timepair(id,time));
	}
	else
		itr->second +=time;
#endif
}

void CLogFile::addTotalTime(string id, time_type in_c1, time_type in_c2, bool is_print)
{
#if LOG_TIME
	//double ela_time = (double)(in_c2 - in_c1) * 1000 / dff;
	double ela_time = elaplsedTime(in_c1, in_c2);

	addTotalTime(id, ela_time, is_print);
#endif
}

void CLogFile::addTotalNum(string id, long in_num, bool is_print)
{
#if LOG_TIME
	if (is_print)
	{
		cout << id << "\t" << in_num <<"\n"; 
	}	

	numbermap::iterator itr = total_num_map.find(id);

	if (itr == total_num_map.end())
	{
		total_num_map.insert(numberpair(id,in_num));
	}
	else
		itr->second +=in_num;
#endif
}

void CLogFile::printTotal()
{
	timemap::iterator itr = total_time_map.begin();
	while(itr != total_time_map.end())
	{
		m_file<< itr->first <<"\t" <<itr->second <<"\n";
		itr++;
	}
	
	numbermap::iterator itr_num = total_num_map.begin();
	while(itr_num != total_num_map.end())
	{
		m_file<< itr_num->first <<"\t" <<itr_num->second <<"\n";
		itr_num++;
	}
}

const string scanKernelName = "scan_kernel_time";
const string lookupName="lookup_kernel_time";
const string extendName="extend_kernel_time";

string argstime[] = {"Total Time",
	"Before search Prepare_time",
	"total search_time",
	"Total Traceback Time",
	"print_time",
	
	"each prepare time",
	"Total PrelimSearch Time",
	"PreliminarySearch Time",
	"Traceback stage time",

	"aux_struct->WordFinder Time",
	"aux_struct->GetGappedScore",
	"Hits extend time",

	"Scan CPU -> GPU Memory Time",
	scanKernelName,
	lookupName,
	extendName,
	//"Scan Kernel Time",
	//"LookUpTableHash Time v1",
	//"kernel_s_BlastNaExtend",
	//"kernel_s_BlastNaExtend_withoutHash Time v1",
	//"kernel_s_BlastSmallExtend Time",
	"GPU->CPU memory Time"};

const int argstime_num = 17;

string argnum[] ={"lookup_hits",
	"good_init_extends",
	"extensions",
	"good_extensions",

	"init_extends",

	"Final hits"};

const int argnum_num = 6;

void CLogFile::printTotalNameBySteps()
{
	for (int i = 0; i < argstime_num; i++)
	{
		timemap::iterator itr = total_time_map.find(argstime[i]);
		if (itr != total_time_map.end())
		{
			m_file << itr->first<<"\t";
		}
	}
	for (int i = 0; i < argnum_num; i++)
	{
		numbermap::iterator itr = total_num_map.find(argnum[i]);
		if (itr != total_num_map.end())
		{
			m_file << itr->first<<"\t";
		}
	}
	m_file << "\n";
}

void CLogFile::printTotalBySteps()
{
	for (int i = 0; i < argstime_num; i++)
	{
		timemap::iterator itr = total_time_map.find(argstime[i]);
		if (itr != total_time_map.end())
		{
			m_file << itr->second<<"\t";
		}
	}
	for (int i = 0; i < argnum_num; i++)
	{
		numbermap::iterator itr = total_num_map.find(argnum[i]);
		if (itr != total_num_map.end())
		{
			m_file << itr->second<<"\t";
		}
	}
	m_file << "\n";
}

void CLogFile::reset()
{
	for (int i = 0; i < argstime_num; i++)
	{
		timemap::iterator itr = total_time_map.find(argstime[i]);
		if (itr != total_time_map.end())
		{
			itr->second = 0;
		}
	}
	for (int i = 0; i < argnum_num; i++)
	{
		numbermap::iterator itr = total_num_map.find(argnum[i]);
		if (itr != total_num_map.end())
		{
			itr->second = 0;
		}
	}
}

CLogFile& slogfile = CLogFile::instance();