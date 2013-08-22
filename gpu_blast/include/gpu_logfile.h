#ifndef __GPU_LOGFILE_H__
#define __GPU_LOGFILE_H__

#ifdef WIN32
#include <Windows.h>
#else
#include <sys/timeb.h>
#define __int64 long long 
#endif


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
	__int64 Start();
	__int64 NewStart(bool is_log = false);
	//end timer
	__int64 End();
	__int64 NewEnd(bool is_log = false);

	//get elaplsed time
	double elaplsedTime();

	// get elaplsed Time with c1 and c2
	double elaplsedTime(__int64 in_c1, __int64 in_c2);

	double printElaplsedTime(string id, __int64 in_c1, __int64 in_c2);

	void KernelStart();
	void KernelEnd();
	double KernelElaplsedTime();
	
	void addTotalTime(string id, double time, bool is_print =true);
	void addTotalTime(string id, __int64 in_c1, __int64 in_c2, bool is_print= true);
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
	__int64  c1, c2;
	float m_eptime;

	timemap total_time_map;
	numbermap total_num_map;
};

extern CLogFile& slogfile;
#endif