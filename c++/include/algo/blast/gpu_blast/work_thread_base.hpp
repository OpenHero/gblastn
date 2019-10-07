#ifndef __THREAD_WORK__
#define __THREAD_WORK__

#include <corelib/ncbistl.hpp>

#ifdef _MSC_VER
#include <Windows.h>
typedef  HANDLE ThreadHandle;
typedef unsigned RETURN_INT;
#endif // _MSC_VER

#ifdef __linux__
#include <pthread.h>
typedef  pthread_t ThreadHandle;
typedef void* RETURN_INT;
#endif

BEGIN_NCBI_SCOPE
BEGIN_SCOPE(blast)

class WorkThreadBase
{
public:
	WorkThreadBase();
	virtual ~WorkThreadBase();

	int start();
	int join();
	int detach();
	void stop();
	ThreadHandle self();

	virtual RETURN_INT run() = 0;

	int        m_running;

private:
	unsigned int	m_tid;
	ThreadHandle		m_thandle;

	int        m_detached;
};
END_SCOPE(blast)
END_NCBI_SCOPE

#endif //__THREAD_WORK__
