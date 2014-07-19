#include <algo/blast/gpu_blast/thread_work_queue.hpp>

#ifdef _MSC_VER
#define MS_VC_EXCEPTION 0x406D1388

#pragma pack(push,8)
typedef struct tagTHREADNAME_INFO
{
	DWORD dwType; // Must be 0x1000.
	LPCSTR szName; // Pointer to name (in user addr space).
	DWORD dwThreadID; // Thread ID (-1=caller thread).
	DWORD dwFlags; // Reserved for future use, must be zero.
} THREADNAME_INFO;
#pragma pack(pop)

static void SetThreadName( DWORD dwThreadID, char* threadName)
{
	//Sleep(10);
	THREADNAME_INFO info;
	info.dwType = 0x1000;
	info.szName = threadName;
	info.dwThreadID = dwThreadID;
	info.dwFlags = 0;

	__try
	{
		RaiseException( MS_VC_EXCEPTION, 0, sizeof(info)/sizeof(ULONG_PTR), (ULONG_PTR*)&info );
	}
	__except(EXCEPTION_EXECUTE_HANDLER)
	{
		;
	}
}
#endif

#if _LINUX
#include<sys/prctl.h>

static void SetThreadName( unsigned long dwThreadID, char* threadName)
{
	prctl(PR_SET_NAME,(unsigned long)threadName);
}
#endif


unsigned long ThreadLock::GetCurrentThreadID()
{
	unsigned long p_thread_id = 0;
#ifdef _MSC_VER
	p_thread_id = GetCurrentThreadId();
#else
	p_thread_id = pthread_self();
#endif

	return p_thread_id;
};

void ThreadLock::SetCurrentThreadName(unsigned long thread_id, string name)
{
	char* threadName = (char*)name.c_str();
	SetThreadName(thread_id, threadName);
}

void ThreadLock::SectionLock()
{
#ifdef _LINUX
	pthread_mutex_lock(&mutex_lock);
#endif
#ifdef _MSC_VER						  
	EnterCriticalSection(&mutex_lock);
#endif
}

void ThreadLock::SectionUnlock()
{
#ifdef _LINUX
	pthread_mutex_unlock(&mutex_lock);
#endif
#ifdef _MSC_VER
	LeaveCriticalSection(&mutex_lock);
#endif
}


#ifdef _LINUX
pthread_mutex_t & ThreadLock::GetMutex() { return mutex_lock;};
#endif

#ifdef _MSC_VER 
CRITICAL_SECTION  & ThreadLock::GetMutex() { return mutex_lock;};
#endif

void ThreadLock::InitMutexLock()
{
#ifdef _LINUX
	pthread_mutex_init(&mutex_lock, NULL);
#endif

#ifdef _MSC_VER
	InitializeCriticalSection(&mutex_lock);
#endif
};

void ThreadLock::DeleteMutexLock()
{
#ifdef _LINUX
	pthread_mutex_destroy(&mutex_lock);
#endif
#ifdef _MSC_VER
	DeleteCriticalSection(&mutex_lock);
#endif
};
