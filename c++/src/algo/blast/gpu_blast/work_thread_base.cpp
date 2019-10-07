#include <algo/blast/gpu_blast/work_thread_base.hpp>

#ifdef _MSC_VER
#include <process.h>
#endif

BEGIN_NCBI_SCOPE
BEGIN_SCOPE(blast)

#ifdef linux__
static void* runThread(void* arg)
{
	return ((WorkThreadBase*)arg)->run();
}
#endif // linux__

#ifdef _MSC_VER
unsigned int __stdcall runThread( void* arg )
{
	return 	((WorkThreadBase*)arg)->run();
}
#endif // _MSC_VER

WorkThreadBase::WorkThreadBase() : m_tid(0), m_running(0), m_detached(0) {}

WorkThreadBase::~WorkThreadBase()
{
	if (m_running == 1) {
#ifdef linux__
		pthread_cancel(m_thandle);
#endif // linux__
#ifdef _MSC_VER
		TerminateThread(m_thandle, 0);
#endif // _MSC_VER
	}
	if (m_running == 1 && m_detached == 0) {
		detach();
	}
}

int WorkThreadBase::start()
{
	int result = 0;
#ifdef _MSC_VER	
	m_thandle = (HANDLE)_beginthreadex(NULL, 0, runThread, this, 0, &m_tid);
	if (0== m_thandle)
	{
		result = -1;
	}
#endif
#ifdef linux__
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);
	result = pthread_create(&m_thandle, &attr, runThread, this);
#endif //linux__
	if (result == 0) {
		m_running = 1;
	}
	return result;
}

int WorkThreadBase::join()
{
	int result = -1;
	if (m_running == 1) {
#ifdef _MSC_VER
		if(WaitForSingleObject(m_thandle, INFINITE) == WAIT_OBJECT_0)
		{
			DWORD status;
			if(GetExitCodeThread(m_thandle, &status) &&	status != DWORD(STILL_ACTIVE))
			{
				if(TRUE == CloseHandle(m_thandle))
				{
					m_thandle = NULL;
					result = 0;
				}
			}
		}
#endif // _MSC_VER
#ifdef linux__
		result = pthread_join(m_thandle, NULL);
#endif // linux__

		if (result == 0) {
			m_detached = 0;
		}
	}
	return result;
}

int WorkThreadBase::detach()
{
	int result = -1;
	if (m_running == 1 && m_detached == 0) {
#ifdef _MSC_VER
		if(TRUE == CloseHandle(m_thandle))
			result = 0;
#endif

#ifdef linux__
		result = pthread_detach(m_thandle);
#endif // linux__
		if (result == 0) {
			m_detached = 1;
		}
	}
	return result;
}
void WorkThreadBase::stop(){
	m_running = 0;
}

ThreadHandle WorkThreadBase::self() {
	return m_thandle;
}

END_SCOPE(blast)
	END_NCBI_SCOPE