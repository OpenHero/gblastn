#ifndef CONNECT_EXT___NCBI_IFCONF__H
#define CONNECT_EXT___NCBI_IFCONF__H

/* $Id: ncbi_ifconf.h 371155 2012-08-06 15:52:52Z lavr $
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
 *   Get host IP and related network configuration information
 *
 *   UNIX only!!!
 *
 */

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#ifndef   INADDR_BROADCAST
#  define INADDR_BROADCAST  ((unsigned int)(-1))
#endif
#ifndef   INADDR_NONE
#  define INADDR_NONE       INADDR_BROADCAST
#endif
#ifndef   INADDR_ANY
#  define INADDR_ANY        0
#endif


#ifdef __cplusplus
extern "C" {
#endif


typedef struct {
    unsigned int address;       /* Primary address, network byte order(b.o.) */
    unsigned int netmask;       /* Primary netmask, network byte order       */
    unsigned int broadcast;     /* Primary broadcast address, network b.o.   */
    int          nifs;          /* Number of network interfaces detected     */
    int          sifs;          /* Number of network interfaces skipped      */
    size_t       mtu;           /* MTU if known for the returned address     */
} SNcbiIfConf;


/* Fill out parameters of primary (first) network interface (NIF)
 * that also has "flags" (e.g. IFF_MULTICAST) set on it, and for which
 * socket "s" was created.  "s" must be >= 0 for the call to work.
 * Return non-zero if at least one NIF has been found;  0 otherwise with
 * "errno" indicating the last OS error condition during the search.
 *
 * NOTE:  Addresses returned are in network byte order, whilst INADDR_*
 * constants are always in host byte order [but by the virtue of values,
 * INADDR_NONE and INADDR_ANY are preserved across representations;
 * but beware of INADDR_LOOPBACK!].
 *
 * This call skips all non-running/non-IP NIFs, or those having private or
 * loopback flags set, or otherwise having flags and/or netmask unobtainable.
 *
 * In case of a non-zero return, NIF information returned may contain:
 * INADDR_NONE as "address", if no NIF matching "flags" has been found;
 * INADDR_LOOPBACK as "address", if only loopback NIF has been found;
 * but in either case "netmask" is guaranteed to have INADDR_ANY
 * (may also want to check "errno" for more information).
 * "Broadcast" is only set for a found NIF that has both "address" and
 * "netmask" distinct from INADDR_NONE and INADDR_ANY, respectively.
 *
 * "nifs" and "sifs" contain the number of NIFs seen during the call
 * (and are not necessarily the total number of interfaces on the machine
 * unless the call finished with no matches found; hence, all NIFs probed).
 */
extern int/*bool*/ NcbiGetHostIfConfEx(SNcbiIfConf* c,
                                       int/*socket*/ s, int/*ifflag*/ flag);


/* Stream IP socket will be created and closed internally to obtain
 * NIF information.  No special flags will be selected.
 * @sa
 *  NcbiGetHostIfConfEx
 */
extern int/*bool*/ NcbiGetHostIfConf(SNcbiIfConf* c);


/* Equivalent of calling NcbiGetHostIfConf() and if successful,
 * printing out "address" field from NIF information structure.
 * Return "buf" on success, 0 on error.
 */
extern char* NcbiGetHostIP(char* buf, size_t bufsize);


#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CONNECT_EXT___NCBI_IFCONF__H */
