/* $Id: ncbi_ifconf.c 371160 2012-08-06 15:55:15Z lavr $
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

#include "../ncbi_assert.h"
#include <connect/ext/ncbi_ifconf.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>
#ifdef HAVE_SYS_SOCKIO_H
#  include <sys/sockio.h>
#endif /*HAVE_SYS_SOCKIO_H*/
#include <net/if.h>


#if defined(HAVE_SOCKLEN_T)  ||  defined(_SOCKLEN_T)
typedef socklen_t  SOCK_socklen_t;
#else
typedef int        SOCK_socklen_t;
#endif /*HAVE_SOCKLEN_T || _SOCKLEN_T*/

#ifdef NCBI_OS_SOLARIS
/* Per: man 7 if_tcp */
#  define ifr_mtu ifr_metric
#endif /*NCBI_OS_SOLARIS*/


extern int/*bool*/ NcbiGetHostIfConfEx(SNcbiIfConf* c, int s, int flag)
{
    unsigned int a, b;
    struct ifconf ifc;
    struct ifreq ifr;
    char *buf, *p;
    size_t size;
    size_t mtu;
    int n, m;

    errno = 0;
    if (!c  ||  s < 0)
        return 0;

    for (size = 1024;  ;  size += 1024) {
        if (!(buf = (char*) calloc(1, size)))
            return 0;

        ifc.ifc_len = (SOCK_socklen_t) size;
        ifc.ifc_buf = buf;
        if ((n = ioctl(s, SIOCGIFCONF, &ifc)) >= 0
            &&  ifc.ifc_len + 2*sizeof(ifr) < size) {
            break;
        }

        free(buf);
        if (n < 0  &&  errno != EINVAL)
            return 0;

        if (size > 100000) {
            errno = E2BIG;
            return 0;
        }
    }

    mtu = 0;
    size = 0;
    errno = 0;
    n = m = 0;
    a = INADDR_NONE;       /* Host byte order */
    b = INADDR_ANY;        /* Host byte order */
    for (p = buf;  p < buf + ifc.ifc_len;  p += size) {
        unsigned int ip;

        memcpy(&ifr, p, sizeof(ifr));
#ifdef _SIZEOF_ADDR_IFREQ
        size = _SIZEOF_ADDR_IFREQ(ifr);
#else
        size = sizeof(ifr);
#endif /*_SIZEOF_ADDR_IFREQ*/

        if (ioctl(s, SIOCGIFADDR, &ifr) < 0)
            continue;

        n++;
        if (ifr.ifr_addr.sa_family != AF_INET) {
            m++;
            continue;
        }

        ip = ntohl(((struct sockaddr_in*) &ifr.ifr_addr)->sin_addr.s_addr);
        if (ip == INADDR_NONE  ||  ip == INADDR_LOOPBACK) {
            if (ip != INADDR_LOOPBACK)
                m++;
            else
                a = ip;
            continue;
        }

        if (ioctl(s, SIOCGIFFLAGS, &ifr) < 0)
            continue;

        if ((ifr.ifr_flags & IFF_LOOPBACK)  ||
#ifdef IFF_PRIVATE
            (ifr.ifr_flags & IFF_PRIVATE)   ||
#endif /*IFF_PRIVATE*/
            !(ifr.ifr_flags & IFF_UP)       ||
            !(ifr.ifr_flags & IFF_RUNNING)  ||
            (flag && !(ifr.ifr_flags & flag)))
            continue;

        if (ioctl(s, SIOCGIFNETMASK, &ifr) < 0)
            continue;

        b = ntohl(((struct sockaddr_in*) &ifr.ifr_addr)->sin_addr.s_addr);
        if (b != INADDR_ANY) {
            a = ip;
#ifdef SIOCGIFMTU
            if (ioctl(s, SIOCGIFMTU, &ifr) >= 0)
                mtu = (size_t)(ifr.ifr_mtu > 0 ? ifr.ifr_mtu : 0);
#endif /*SIOCGIFMTU*/
            break;
        }
    }

    free(buf);

    c->address   = htonl(a);
    c->netmask   = htonl(b);
    c->broadcast = ((a == INADDR_NONE  ||  b == INADDR_ANY) ? 0 :
                    (c->address & c->netmask) | ~c->netmask);
    c->nifs = n;
    c->sifs = m;
    c->mtu  = mtu;

    return 1;
}


extern int/*bool*/ NcbiGetHostIfConf(SNcbiIfConf* c)
{
    int s = socket(AF_INET, SOCK_STREAM, 0);
    int/*bool*/ retval = NcbiGetHostIfConfEx(c, s, 0);
    int x_errno = errno;
    close(s);
    errno = x_errno;
    return retval;
}


extern char* NcbiGetHostIP(char* buf, size_t bufsize)
{
    char str[32];
    SNcbiIfConf c;
    const unsigned char* b = (const unsigned char*) &c.address;

    assert(buf && bufsize > 0);

    if ( NcbiGetHostIfConf(&c) ) {
        verify(sprintf(str, "%u.%u.%u.%u",
                       (unsigned) b[0], (unsigned) b[1],
                       (unsigned) b[2], (unsigned) b[3]) > 0);
        assert(strlen(str) < sizeof(str));
        if (strlen(str) < bufsize) {
            strcpy(buf, str);
            return buf;
        }
    }
    buf[0] = '\0';
    return 0;
}
