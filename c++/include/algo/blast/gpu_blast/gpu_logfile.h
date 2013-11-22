#ifndef __GPU_LOGFILE_H__
#define __GPU_LOGFILE_H__

#ifdef WIN32
#include <Windows.h>
#define time_type __int64
#else
#include <sys/timeb.h>
#define __int64 long long
#define time_type long long 
#endif

#define LOG_TIME 0
#define TIME_GPU 0

#include <fstream>
#include <string>
#include <map>


using namespace std;

typedef map<string, double> timemap;
typedef pair<string, double> timepair;

typedef map<string, long> numbermap;
typedef pair<string, long> numberpair;

class StopWatchInterface;

class CLogFile
{
	CLogFile();
	CLogFile(const CLogFile&);
	CLogFile& operator =(CLogFile);
	~CLogFile();

public:
	static CLogFile & instance()
	{
		static CLogFile s;
		return s;
	}
	
	fstream m_file;

	//start timer
	time_type Start();
	time_type NewStart(bool is_log = false);
	//end timer
	time_type End();
	time_type NewEnd(bool is_log = false);

	//get elaplsed time
	double elaplsedTime();

	// get elaplsed Time with c1 and c2
	double elaplsedTime(time_type in_c1, time_type in_c2);

	double printElaplsedTime(string id, time_type in_c1, time_type in_c2);

	void KernelStart();
	void KernelEnd();
	double KernelElaplsedTime();
	
	void addTotalTime(string id, double time, bool is_print =true);
	void addTotalTime(string id, time_type in_c1, time_type in_c2, bool is_print= true);
	void printTotal();

	void addTotalNum(string id, long in_num, bool is_print = true);

	void printTotalNameBySteps();
	void printTotalBySteps();

	void reset();
private:

	//*************************

	StopWatchInterface *timer;
#ifdef WIN32

	LARGE_INTEGER  large_interger;  
#endif
	

	double dff;  
	time_type  c1, c2;
	float m_eptime;

	timemap total_time_map;
	numbermap total_num_map;
};

extern CLogFile& slogfile;
#endif