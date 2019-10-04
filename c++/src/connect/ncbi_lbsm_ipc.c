/* $Id: ncbi_lbsm_ipc.c 368257 2012-07-05 16:25:00Z lavr $
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
 * Author:  Anton Lavrentiev
 *
 * File Description:
 *   Implementation of the LBSM client-server data exchange API
 *   with the use of SYSV IPC
 *
 *   UNIX only !!!
 *
 */

#include "ncbi_lbsm_ipc.h"
#include "ncbi_priv.h"
#include <errno.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define NCBI_USE_ERRCODE_X   Connect_LBSM


static int         s_Muxid        =  -1;
static int         s_Shmid[2]     = {-1, -1};
static void*       s_Shmem[2]     = { 0,  0};
static TNCBI_Size  s_ShmemSize[2] = { 0,  0};
static const int   k_ShmemKey [2] = {LBSM_SHMEM_KEY_1, LBSM_SHMEM_KEY_2};
static int/*bool*/ s_NoUndo   [4] = {0,0,0,0}; /* for semnums 1, 2, 3, and 4 */


/* We have 5 semaphores in the array for shared memory access control:
 * semnum
 *    0      0 when the daemon is not running, 1 when it is
 *    1      0 when memory is not pre-locked for writing, 1 when it is
 *               however update happens only when semnum(2) reaches zero
 *    2      number of current shared memory accessors, i.e. those
 *               passed via semnum(1) (and perhaps semnum(2))
 *    3      same as 1 and 2 but for the copy of the shmem segment
 *    4
 * All readers are blocked by the semnum[1] > 0.
 * Writers always wait for the semnum[2] == 0 before proceeding.
 * [1] means access to the semnum 1 or 3 (1st sem in the block);
 * [2] means access to the semnum 2 or 4 (2nd sem in the block).
 */


/* Non-blocking decrement of a specified [sem] */
static int/*syscall*/ s_Shmem_Unlock(int which, int sem)
{
    int semnum = (which << 1) + sem;
    struct sembuf unlock;

    unlock.sem_num =  semnum;
    unlock.sem_op  = -1;
    unlock.sem_flg =  IPC_NOWAIT | (s_NoUndo[semnum - 1] ? 0 : SEM_UNDO);
    if (semop(s_Muxid, &unlock, 1) >= 0)
        return 0;
    return errno == EAGAIN ? 0 : -1;
}


/* For 'which' block: check specified [sem] first, then continue with [2]++ */
static int/*syscall*/ s_Shmem_Lock(int which, int sem, int no_wait)
{
    int semnum = (which << 1) + sem;
    int accsem = (which << 1) + 2;
    int/*bool*/ no_undo = 0;
    struct sembuf lock[2];
    int i = 0;

    for (;;) {
        lock[0].sem_num = semnum;
        lock[0].sem_op  = 0; /* precondition:  [sem] == 0 */
        lock[0].sem_flg = no_wait ? IPC_NOWAIT : 0;
        lock[1].sem_num = accsem;
        lock[1].sem_op  = 1; /* postcondition: [2]++      */
        lock[1].sem_flg = no_undo ? 0 : SEM_UNDO;

        if (semop(s_Muxid, lock, sizeof(lock)/sizeof(lock[0])) >= 0) {
            s_NoUndo[accsem - 1] = no_undo;
            return 0;
        }
        if (i)
            break;
        i = errno;
        if (i == ENOSPC) {
            CORE_LOGF_X(7, eLOG_Warning,
                        ("LBSM %c-locking[%d] w/o undo",
                         "RW"[sem > 1], which + 1));
            no_undo = 1;
            continue;
        }
        if (i == EINTR)
            continue;
        if (no_wait  ||  i != ENOMEM)
            break;
#ifdef LBSM_DEBUG
        CORE_LOGF(eLOG_Trace,
                  ("LBSM %c-locking[%d] wait(ENOMEM)",
                   "RW"[sem > 1], which + 1));
#endif /*LBSM_DEBUG*/
        sleep(1);
    }
    return -1;
}


static int/*bool*/ s_Shmem_RUnlock(int which)
{
#ifdef LBSM_DEBUG
    CORE_LOGF(eLOG_Trace, ("LBSM R-lock[%d] release", which + 1));
#endif /*LBSM_DEBUG*/
    /* Decrement number of accessors, [2] */
    return s_Shmem_Unlock(which, 2) == 0;
}


/* Return locked block number {0, 1}, or -1 on error */
static int/*tri-state*/ s_Shmem_RLock(int which)
{
#ifdef LBSM_DEBUG
    CORE_LOG(eLOG_Trace, "LBSM R-lock acquire (%d)", which + 1);
#endif /*LBSM_DEBUG*/
    /* For block 1 then block 0: check [1] first, then continue with [2]++ */
    if (s_Shmem_Lock(which, 1, which/*1:no-wait,0:wait*/) == 0)
        return which/*block locked*/;
    return !which  ||  errno == EINVAL ? -1 : s_Shmem_Lock(0, 1, 0/*wait*/);
}


static int/*bool*/ s_Shmem_WUnlock(int which)
{
    /* Decrement number of accessors [2]; should become 0 after that */
    int retval1 = s_Shmem_Unlock(which, 2);
    /* Open read semaphore [1] */
    int retval2 = s_Shmem_Unlock(which, 1);
#ifdef LBSM_DEBUG
    CORE_LOGF(eLOG_Trace, ("LBSM W-lock[%d] release", which + 1));
#endif /*LBSM_DEBUG*/
    return retval1 == 0  &&  retval2 == 0;
}


static int/*tri-state-bool, inverted*/ s_Shmem_TryWLock(int which)
{
    int semnum = (which << 1) | 1;
    int/*bool*/ no_undo = 0;
    struct sembuf rwlock[2];
    int i = 0;

    for (;;) {
        rwlock[0].sem_num = semnum;
        rwlock[0].sem_op  = 0; /* precondition:  [1] == 0 */
        rwlock[0].sem_flg = 0;
        rwlock[1].sem_num = semnum;
        rwlock[1].sem_op  = 1; /* postcondition: [1] == 1 */
        rwlock[1].sem_flg = no_undo ? 0 : SEM_UNDO;

        if (semop(s_Muxid, rwlock, sizeof(rwlock)/sizeof(rwlock[0])) >= 0) {
            /* No new read accesses are allowed after this point */
            s_NoUndo[semnum - 1] = no_undo;
            /* Look at [2] first, then continue with it ([2]++) */
            if (s_Shmem_Lock(which, 2, 0/*wait*/) != 0)
                return 1/*partially locked*/;
            return 0/*okay*/;
        }
        if (i)
            break;
        i = errno;
        if (i == ENOSPC) {
            CORE_LOGF_X(8, eLOG_Warning,
                        ("LBSM PreW-locking[%d] w/o undo", which + 1));
            no_undo = 1;
            continue;
        }
        if (i == EINTR)
            continue;
        if (i != ENOMEM)
            break;
#ifdef LBSM_DEBUG
        CORE_LOGF(eLOG_Trace,
                  ("LBSM PreW-lock[%d] wait(ENOMEM)", which + 1));
#endif/*LBSM_DEBUG*/
        sleep(1);
    }
    return -1/*not locked*/;
}


/* Return non-zero if successful;  return zero on failure (no lock acquired).
 * If "wait" passed greater than zero do not attempt to assassinate a contender
 * (try to identify and print its PID anyways).  Otherwise, having killed
 * the contender, try to re-acquire the lock (without.
 */
static int/*bool*/ s_Shmem_WLock(int which, int/*bool*/ wait)
{
    static union semun dummy;
    int locked;
    int pid;

#ifdef LBSM_DEBUG
    CORE_LOGF(eLOG_Trace,
              ("LBSM W-lock[%d] acquire%s", which + 1, wait ? " w/wait" : ""));
#endif /*LBSM_DEBUG*/

    if ((locked = s_Shmem_TryWLock(which)) == 0)
        return 1/*success*/;

    if (locked < 0) {
        /* even [1] was not successfully locked, so we have either
         *   a/ a hanging writer, or
         *   b/ a hanging old reader (which doesn't change [2] in block 0 only)
         * In either case, we can try to obtain PID of the offender.
         */
        if ((pid = semctl(s_Muxid, (which << 1) | 1, GETPID, dummy)) > 0) {
            int self   = (pid_t) pid == getpid();
            int other  = !self  &&  (kill(pid, 0) == 0  ||  errno == EPERM);
            int killed = 0;
            if (!wait) {
                if (other  &&  kill(pid, SIGTERM) == 0) {
                    CORE_LOGF_X(17, eLOG_Warning,
                                ("Terminating PID %lu", (long) pid));
                    sleep(1); /* let them catch SIGTERM and exit gracefully */
                    kill(pid, SIGKILL);
                    killed = 1;
                } else {
                    CORE_LOGF_X(18, eLOG_Warning,
                                ("Unable to kill PID %lu", (long) pid));
                }
            }
            CORE_LOGF_X(19, eLOG_Warning,
                        ("LBSM lock[%d] %s revoked from PID %lu (%s)",
                         which + 1, killed ? "is being" : "has to be",
                         (long) pid, self  ? "self" :
                         other ? (killed ? "killed" : "hanging") : "zombie"));
        } else if (pid < 0)
            return 0/*severe failure: most likely removed IPC id*/;
    } else {
        pid = 0;
        /* [1] was free (now locked) but [2] was taken by someone else,
         * we have a hanging reader, and no additional info can be obtained.
         */
        if (!wait) {
            union semun arg;
            arg.val = 1;
            if (semctl(s_Muxid, (which << 1) + 2, SETVAL, arg) < 0) {
                CORE_LOGF_ERRNO_X(9, eLOG_Error, errno,
                                  ("Unable to adjust LBSM access count[%d]",
                                   which + 1));
            }
            wait = 1/*this makes us undo [1] and fail*/;
        }
        if (wait) {
            int x_errno = errno;
            s_Shmem_Unlock(which, 1);
            errno = x_errno;
        }
    }

    if (!pid) {
        int  val;
        char num[32];
        if ((val = semctl(s_Muxid, (which << 1) + 2, GETVAL, dummy)) > 1)
            sprintf(num, "%d", val);
        else
            strcpy(num, "A");
        CORE_LOGF_X(20, eLOG_Warning,
                    ("%s hanging reader%s in LBSM shmem[%d]%s",
                     num, val > 1 ? "s" : "", which + 1, wait ? "" :
                     locked < 0 ? ", revoking lock" : ", lock revoked"));
    }

    if (wait)
        return 0/*failure*/;

    if (locked > 0)
        return 1/*success: good to go*/;

#ifdef LBSM_DEBUG
    CORE_LOGF(eLOG_Trace,
              ("LBSM W-lock[%d] re-acquire", which + 1));
#endif /*LBSM_DEBUG*/
    return s_Shmem_WLock(which, 0/*no wait*/);
}


static HEAP s_Shmem_Attach(int which)
{
    void* shmem = 0; /*dummy init for compiler not to complain*/
    int   shmid;
    HEAP  heap;

#ifdef LBSM_DEBUG
    CORE_LOGF(eLOG_Trace, ("LBSM shmem[%d] attaching", which + 1));
#endif /*LBSM_DEBUG*/
    if ((shmid = shmget(k_ShmemKey[which], 0, 0)) >= 0  &&
        (shmid == s_Shmid[which]  ||
         ((shmem = shmat(shmid, 0, SHM_RDONLY))  &&  shmem != (void*)(-1)))) {
        if (shmid != s_Shmid[which]) {
            struct shmid_ds shm_ds;
#ifdef LBSM_DEBUG
            CORE_LOGF(eLOG_Trace, ("LBSM shmem[%d] attached", which + 1));
#endif /*LBSM_DEBUG*/
            s_Shmid[which] = shmid;
            if (s_Shmem[which])
                shmdt(s_Shmem[which]);
            s_Shmem[which] = shmem;
            s_ShmemSize[which] = (shmctl(shmid, IPC_STAT, &shm_ds) < 0
                                  ? 0 : shm_ds.shm_segsz);
        }
        heap = (s_ShmemSize[which]
                ? HEAP_AttachFast(s_Shmem[which], s_ShmemSize[which], which+1)
                : HEAP_Attach    (s_Shmem[which], which+1));
    } else
        heap = 0;

    return heap;
}


HEAP LBSM_Shmem_Attach(int/*bool*/ fallback)
{
    int  which;
    HEAP heap;

#ifdef LBSM_DEBUG
    CORE_LOG(eLOG_Trace, "LBSM ATTACHING%s", fallback ? " (fallback)" : "");
#endif /*LBSM_DEBUG*/
    if ((which = s_Shmem_RLock(!fallback)) < 0) {
        CORE_LOG_ERRNO_X(10, eLOG_Warning, errno,
                         "Cannot lock LBSM shmem to attach");
        return 0;
    }
#ifdef LBSM_DEBUG
    CORE_LOGF(eLOG_Trace, ("LBSM R-lock[%d] acquired", which + 1));
#endif /*LBSM_DEBUG*/
    if (!(heap = s_Shmem_Attach(which))) {
        int/*bool*/ attached = s_Shmem[which] != 0 ? 1/*true*/ : 0/*false*/;
        s_Shmem_RUnlock(which);
        CORE_LOGF_ERRNO_X(11, eLOG_Error, errno,
                          ("Cannot %s LBSM shmem[%d]",
                           attached ? "access" : "attach", which + 1));
    }
#ifdef LBSM_DEBUG
    else {
        CORE_LOGF(eLOG_Trace,
                  ("LBSM heap[%p, %p, %d] attached",
                   heap, HEAP_Base(heap), which + 1));
        assert(HEAP_Serial(heap) == which + 1);
    }
#endif /*LBSM_DEBUG*/
    if (s_Shmem[which = !which]) {
#ifdef LBSM_DEBUG
        CORE_LOGF(eLOG_Trace, ("LBSM shmem[%d] detached", which + 1));
#endif /*LBSM_DEBUG*/
        shmdt(s_Shmem[which]);
        s_Shmem[which] =  0;
        s_Shmid[which] = -1;
    } else
        assert(s_Shmid[which] < 0);
    s_ShmemSize[which] =  0;
    return heap;
}


void LBSM_Shmem_Detach(HEAP heap)
{
    int which;

    if (!heap)
        return;
    which = HEAP_Serial(heap);
#ifdef LBSM_DEBUG
    CORE_LOGF(eLOG_Trace, ("LBSM shmem[%d] detaching", which));
#endif /*LBSM_DEBUG*/
    if (which != 1  &&  which != 2) {
        CORE_LOGF_X(12, eLOG_Error,
                    ("Bad block number (%d) for LBSM shmem to unlock", which));
    } else
        s_Shmem_RUnlock(which - 1);
    HEAP_Detach(heap);
#ifdef LBSM_DEBUG
    CORE_LOG(eLOG_Trace, "LBSM DETACHED");
#endif /*LBSM_DEBUG*/
}


#ifdef __cplusplus
extern "C" {
    static void* s_LBSM_ExpandHeap(void*, TNCBI_Size, void*);
}
#endif

/*ARGSUSED*/
static void* s_LBSM_ExpandHeap(void* mem, TNCBI_Size newsize, void* arg)
{
    if (mem  &&  newsize)
        return realloc(mem, newsize);
    if (newsize) /* mem     == 0 */
        return malloc(newsize);
    if (mem)     /* newsize == 0 */
        free(mem);
    return 0;
}


HEAP LBSM_Shmem_Create(void)
{
    int/*bool*/ one = 0/*false*/, two = 0/*false*/;
    HEAP heap = 0;
    size_t i;

    for (i = 0;  i < 2;  i++)
        s_Shmid[i] = shmget(k_ShmemKey[i], 0, 0);
    if ((one = (s_Shmid[0] >= 0)) | (two = (s_Shmid[1] >= 0))) {
        CORE_LOGF_X(13, eLOG_Warning,
                    ("Re-creating existing LBSM shmem segment%s %s%s%s",
                     one ^ two ? ""    : "s",
                     one       ? "[1]" : "",
                     one ^ two ? ""    : " and ",
                     two       ? "[2]" : ""));
        LBSM_Shmem_Destroy(0);
    }
    if (!(i = CORE_GetVMPageSize()))
        i = 4096;
    heap = HEAP_Create(0, 0, i, s_LBSM_ExpandHeap, 0);
    return heap;
}


static void s_Shmem_Destroy(int which, pid_t own_pid)
{
    if (s_Shmid[which] < 0) {
        assert(!s_Shmem[which]  &&  !s_ShmemSize[which]);
        return;
    }
    if (s_Shmem[which]) {
        if (shmdt(s_Shmem[which]) < 0) {
            CORE_LOGF_ERRNO_X(14, eLOG_Error, errno,
                              ("Unable to detach LBSM shmem[%d]", which + 1));
        }
        s_Shmem[which] = 0;
    }
    if (own_pid) {
        struct shmid_ds shm_ds;
        if (shmctl(s_Shmid[which], IPC_STAT, &shm_ds) < 0)
            shm_ds.shm_cpid = 0;
        if (shm_ds.shm_cpid != own_pid) {
            if (shm_ds.shm_cpid) {
                CORE_LOGF_X(15, eLOG_Error,
                            ("Not an owner (%lu) to remove LBSM shmem[%d]",
                             (long) shm_ds.shm_cpid, which + 1));
            } else {
                CORE_LOGF_ERRNO(eLOG_Error, errno,
                                ("Unable to stat LBSM shmem[%d]", which + 1));
            }
            own_pid = 0;
        }
    } else
        own_pid = 1;
    if (own_pid  &&  shmctl(s_Shmid[which], IPC_RMID, 0) < 0) {
        CORE_LOGF_ERRNO_X(16, eLOG_Error, errno,
                          ("Unable to remove LBSM shmem[%d]", which + 1));
    }
    s_Shmid[which]     = -1;
    s_ShmemSize[which] =  0;
}


void LBSM_Shmem_Destroy(HEAP heap)
{
    pid_t self = heap ? getpid() : 0;
    int i;

    for (i = 0;  i < 2;  i++)
        s_Shmem_Destroy(i, self);
    HEAP_Destroy(heap);
}


unsigned int LBSM_Shmem_Update(HEAP heap, int/*bool*/ wait)
{
    size_t heapsize = HEAP_Size(heap);
    void*  heapbase = HEAP_Base(heap);
    unsigned int updated = 0;
    int i;

    assert(heapbase  &&  heapsize);
    for (i = 0;  i < 2;  i++) {
        int can_wait = wait  ||  (s_ShmemSize[i]  &&  (!i  ||  updated));
        if (!s_Shmem_WLock(i, can_wait))
            continue;

        /* Update shmem here: strict checks for the first time */
        if (s_ShmemSize[i] != heapsize) {
            int   shmid;
            void* shmem;
            s_Shmem_Destroy(i, s_ShmemSize[i] ? 0 : getpid());
            if ((shmid = shmget(k_ShmemKey[i], heapsize,
                                LBSM_SHM_PROT | IPC_CREAT | IPC_EXCL)) < 0  ||
                !(shmem = shmat(shmid, 0, 0))  ||  shmem == (void*)(-1)) {
                CORE_LOGF_ERRNO_X(22, eLOG_Error, errno,
                                  ("Unable to re-create LBSM shmem[%d]",
                                   i + 1));
                s_Shmem_WUnlock(i);
                return 0/*update failed*/;
            }
            s_Shmid[i] = shmid;
            s_Shmem[i] = shmem;
            s_ShmemSize[i] = heapsize;
        }
        memcpy(s_Shmem[i], heapbase, heapsize);
        if (!s_Shmem_WUnlock(i)) {
            CORE_LOGF_ERRNO_X(21, eLOG_Warning, errno,
                              ("Update failed to unlock shmem[%d]", i + 1));
        }
        updated |= 1 << i;
    }
    return updated;
}


int LBSM_LBSMD(int/*bool*/ check_n_lock)
{
    struct sembuf lock[2];
    int id = semget(LBSM_MUTEX_KEY, check_n_lock ? 5 : 0,
                    check_n_lock ? (IPC_CREAT | LBSM_SEM_PROT) : 0);
    if (id < 0)
        return -1;

    s_Muxid = id;
    /* Check & lock daemon presence: done atomically */
    lock[0].sem_num = 0;
    lock[0].sem_op  = 0;
    lock[0].sem_flg = IPC_NOWAIT;
    lock[1].sem_num = 0;
    lock[1].sem_op  = 1;
    lock[1].sem_flg = SEM_UNDO;
    if (semop(id, lock,
              sizeof(lock)/sizeof(lock[0]) - (check_n_lock ? 0 : 1)) < 0) {
        return 1;
    }
    return 0;
}


/* Daemon use: undaemon > 0; Client use: undaemon == 0 => return LBSMD PID
 *                                       undaemon <  0 => fast return      */
pid_t LBSM_UnLBSMD(int/*bool*/ undaemon)
{
    static union semun dummy;
    pid_t pid = 0;
    int which;

    if (s_Muxid >= 0) {
        if (undaemon <= 0) {
            if (!undaemon) {
                if ((which = s_Shmem_RLock(1)) >= 0) {
                    int shmid = shmget(k_ShmemKey[which], 0, 0);
                    struct shmid_ds shm_ds;
                    if (shmid > 0  &&  shmctl(shmid, IPC_STAT, &shm_ds) == 0)
                        pid = shm_ds.shm_cpid;
                    s_Shmem_RUnlock(which);
                }
            }
            CORE_LOCK_WRITE;
            for (which = 0;  which < 2;  which++) {
                if (s_Shmem[which]) {
#ifdef LBSM_DEBUG
                    CORE_LOGF(eLOG_Trace, ("LBSM shmem[%d] detached",which+1));
#endif /*LBSM_DEBUG*/
                    shmdt(s_Shmem[which]);
                    s_Shmem[which] =  0;
                    s_Shmid[which] = -1;
                } else
                    assert(s_Shmid[which] < 0);
                s_ShmemSize[which] =  0;
            }
            CORE_UNLOCK;
        } else {
            semctl(s_Muxid, 0, IPC_RMID, dummy);
            s_Muxid = -1;
        }
    }
    return pid;
}
