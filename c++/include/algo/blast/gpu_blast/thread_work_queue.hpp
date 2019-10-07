#ifndef __WORK_QUEUE_HPP__
#define __WORK_QUEUE_HPP__

#ifdef _MSC_VER
#include <windows.h>
#else
#include <pthread.h>
#endif

#include <queue>
#include <string>

using namespace std;

//////////////////////////////////////////////////////////////////////////
//thread mutex

class ThreadLock
{
public:
	ThreadLock()
	{
		this->InitMutexLock();
	};
	~ThreadLock()
	{
		this->DeleteMutexLock();
	};

	unsigned long GetCurrentThreadID();
	void SetCurrentThreadName(unsigned long thread_id, string name);  
	void SectionLock();									 
	void SectionUnlock();


#ifdef linux__
	pthread_mutex_t & GetMutex();
#endif

#ifdef _MSC_VER 
	CRITICAL_SECTION  & GetMutex();
#endif

protected:
	void InitMutexLock();

	void DeleteMutexLock();

private:
#ifdef linux__
	pthread_mutex_t   mutex_lock;
#endif

#ifdef _MSC_VER 
	CRITICAL_SECTION  mutex_lock;
#endif
};

class ThreadEvent
{
public:

#ifdef linux__
	ThreadEvent() 
	{
		pthread_cond_init(&m_event, NULL);
	}
	~ThreadEvent()
	{
		pthread_cond_destroy(&m_event);
	}

	void AnounceEvent()
	{
		pthread_cond_signal(&m_event);
	}
	void WaitingSingleEvent(pthread_mutex_t& i_mutex)
	{
		pthread_cond_wait(&m_event, &i_mutex);
	}
protected:
private:
	pthread_cond_t	m_event;
#endif 

#ifdef _MSC_VER
	ThreadEvent() 
	{
		m_event = CreateEvent(NULL, FALSE, FALSE, NULL);
	}
	~ThreadEvent()
	{
		CloseHandle(m_event); 
	}

	void AnounceEvent()
	{
		PulseEvent(m_event);
		//SetEvent(m_event);
	}
	void WaitingSingleEvent()
	{
		WaitForSingleObject(m_event, INFINITE); 
	}
private:
	HANDLE			m_event;
#endif
};

template <typename T> class work_queue
{
	queue<T>		m_queue;
	ThreadLock		m_tlock;
	ThreadEvent		m_event; 

public:
	work_queue() {
	}
	~work_queue() {
	}
	void add(T item) {
		m_tlock.SectionLock();
		m_queue.push(item);
		m_event.AnounceEvent();
		m_tlock.SectionUnlock();
	}

#ifdef linux__
	T remove() {
		m_tlock.SectionLock();
		while (m_queue.size() == 0) {

			m_event.WaitingSingleEvent(m_tlock.GetMutex());

		}
		T item = m_queue.front();
		m_queue.pop();
		m_tlock.SectionUnlock();
		return item;
	} 
#endif

#ifdef _MSC_VER
	T remove() {
		//m_tlock.SectionLock();
		while (m_queue.size() == 0) {	 
			m_event.WaitingSingleEvent();
		}
		T item = m_queue.front();
		m_queue.pop();
		//m_tlock.SectionUnlock();
		return item;
	}
#endif

	int size() {
		m_tlock.SectionLock();
		int size = m_queue.size();
		m_tlock.SectionUnlock();
		return size;
	}
};

#endif //__WORK_QUEUE_HPP__