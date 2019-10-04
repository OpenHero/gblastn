/* $Id: ncbi_localnet.c 371155 2012-08-06 15:52:52Z lavr $
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
 *   Get IP address of CGI client and determine the IP locality
 *
 *   NOTE:  This is an internal NCBI-only API not for export to third parties.
 *
 */

#include "../ncbi_ansi_ext.h"
#include "../ncbi_priv.h"
#include <connect/ncbi_connutil.h>
#if defined(NCBI_OS_UNIX)
#  include <connect/ext/ncbi_ifconf.h>
#elif !defined(INADDR_LOOPBACK)
#  define      INADDR_LOOPBACK  0x1F000001
#endif      /*!INADDR_LOOPBACK*/
#include <connect/ext/ncbi_iprange.h>
#include <connect/ext/ncbi_localnet.h>
#include <assert.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef NCBI_OS_UNIX
#  include <sys/param.h>
#endif /*NCBI_OS_UNIX*/

#define NCBI_USE_ERRCODE_X   Connect_LocalNet


#if defined(NCBI_OS_MSWIN)  &&  !defined(PATH_MAX)
#  ifdef _MAX_PATH
#    define PATH_MAX  _MAX_PATH
#else
#    define PATH_MAX  1024
#  endif /*_MAX_PATH*/
#endif /*NCBI_OS_MSWIN && !PATH_MAX*/

#if PATH_MAX < 256
#  define BUFSIZE  256
#else
#  define BUFSIZE  PATH_MAX
#endif /*PATHMAX<256*/

#if defined(_DEBUG)  &&  !defined(NDEBUG)
/*#  define LOCALNET_DEBUG 1*/
#endif /*_DEBUG && !NDEBUG*/


static int/*bool*/ s_Inited = 0/*false*/;


static SIPRange s_LocalIP[256 + 1] = { { eIPRange_None } };


static SIPRange kLocalIP[] = {
    /* localnet (localhost):  127/8 */
#if defined(IN_CLASSA) && defined(IN_CLASSA_NET) && defined(IN_CLASSA_NSHIFT)
#  if IN_CLASSA_MAX <= IN_LOOPBACKNET
#    error "IN_LOOPBACKNET is out of range"
#  endif /*IN_CLASSA_MAX<=IN_LOOPBACKNET*/
    { eIPRange_Network, IN_LOOPBACKNET << IN_CLASSA_NSHIFT, IN_CLASSA_NET},
#else
    { eIPRange_Network, INADDR_LOOPBACK-1, 0xFF000000 },
#endif /*IN_CLASSA && IN_CLASSA_NET && IN_CLASSA_NSHIFT*/
    /* from assigned IP ranges */
    { eIPRange_Range,   0x820E0800, 0x820E09FF }, /* 130.14.{8|9}.0/24       */
    { eIPRange_Range,   0x820E0B00, 0x820E0CFF }, /* 130.14.1{1|2}.0/24      */
    { eIPRange_Range,   0x820E1400, 0x820E1AFF }, /* 130.14.20.0..27.255     */
    { eIPRange_Range,   0x820E1B40, 0x820E1BFF }, /*  w/o 130.14.27.0/26     */
    { eIPRange_Network, 0x820E1D00, 0xFFFFFF00 }, /* 130.14.29.yyy           */
    { eIPRange_Network, 0xA5700700, 0xFFFFFF00 }, /* 165.112.7.zzz (colo)    */
    /* from private IP ranges */
    { eIPRange_Network, 0x0A0A0000, 0xFFFF0000 }, /* 10.10/16      from cl.A */
    { eIPRange_Network, 0x0A140000, 0xFFFF0000 }, /* 10.20/16      from cl.A */
    { eIPRange_Network, 0xAC100000, 0xFFF00000 }, /* 172.16/12  16 nets cl.B */
    { eIPRange_None }
};


static int/*bool*/ s_OverlapLocalIP(const SIPRange* range, size_t n)
{
    SIPRange r1 = NcbiTrueIPRange(range);
    size_t i;

    for (i = 0;  i < n;  i++) {
        SIPRange r2 = NcbiTrueIPRange(s_LocalIP + i);
        unsigned int a = r1.a > r2.a ? r1.a : r2.a;
        unsigned int b = r1.b < r2.b ? r1.b : r2.b;
        if (a <= b)
            return 1/*true*/;
    }
    return 0/*false*/;
}


static void s_LoadLocalIPs(void)
{
    char buf[PATH_MAX + 1];
    const char* filename = ConnNetInfo_GetValue(0, "LOCAL_IPS",
                                                buf, sizeof(buf) - 1,
                                                "/etc/ncbi/local_ips");
    int lineno;
    size_t n;
    FILE *fp;

    if (filename  &&  strcasecmp(filename, "--HARDCODED--") == 0)
        filename = 0;
    if (!filename  ||  !(fp = fopen(filename, "r"))) {
        if (filename) {
            CORE_LOGF_ERRNO_X(1, errno == ENOENT ? eLOG_Warning : eLOG_Error,
                              errno, ("Cannot load local IP specs from '%s'",
                                      buf));
        }
        CORE_LOG(eLOG_Trace, "Using default local IP specs");
        assert(sizeof(s_LocalIP) >= sizeof(kLocalIP));
        memcpy(s_LocalIP, kLocalIP, sizeof(kLocalIP));
        return;
    }

    CORE_LOGF(eLOG_Trace, ("Loading local IP specs from '%s'", filename));
    memcpy(s_LocalIP, kLocalIP, sizeof(kLocalIP[0]));
    n = 1/*localhost gets always added*/;
    lineno = 0;
    do {
        SIPRange local;
        char* c, *err;
        size_t len;
        if (!fgets(buf, sizeof(buf) - 1, fp))
            break;
        lineno++;
        if (!(len = strcspn(buf, "!#")))
            continue;
        if (buf[len]) {
            buf[len] = '\0';
        } else if (buf[len - 1] == '\n') {
            if (len > 1  &&  buf[len - 2] == '\r')
                len--;
            buf[len - 1] = '\0';
        }
        if (!*(c = buf + strspn(buf, " \t")))
            continue;
        len = strcspn(c, " \t");
        err = c + len;
        if (*err  &&  !*(err += strspn(err, " \t")))
            c[len] = '\0';
        if (*err  ||  !NcbiParseIPRange(&local, c)) {
            if (!*err)
                err = c;
            CORE_LOGF_X(2, eLOG_Error,
                        ("Local IP spec at line %u, '%s' is invalid",
                         lineno, err));
            break;
        }
        if (local.type == eIPRange_None)
            continue;
        if (n >= sizeof(s_LocalIP)/sizeof(s_LocalIP[0])) {
            CORE_LOGF_X(3, eLOG_Error,
                        ("Too many local IP specs, max %u allowed",
                         (unsigned int)(n - 1)));
            break;
        }
        if (s_OverlapLocalIP(&local, n)) {
            CORE_LOGF_X(4, eLOG_Warning,
                        ("Local IP spec at line %u, '%s' overlaps with"
                         " already defined one(s)", lineno, c));
        }
        s_LocalIP[n++] = local;
    } while (!feof(fp));
    fclose(fp);

    CORE_LOGF(eLOG_Trace, ("Done loading local IP specs, %u line%s, %u entr%s",
                           lineno, &"s"[lineno == 1], (unsigned int) n,
                           n == 1 ? "y" : "ies"));
    if (n < sizeof(s_LocalIP)/sizeof(s_LocalIP[0]))
        s_LocalIP[n].type = eIPRange_None;
}


extern void NcbiInitLocalIP(void)
{
    s_Inited = 0;
}


extern int/*bool*/ NcbiIsLocalIP(unsigned int ip)
{
    size_t n;
    if (!s_Inited) {
        CORE_LOCK_WRITE;
        if (!s_Inited) {
            s_LoadLocalIPs();
            s_Inited = 1;
            CORE_UNLOCK;
#ifdef LOCALNET_DEBUG
            for (n = 0;  n < sizeof(s_LocalIP)/sizeof(s_LocalIP[0]);  n++) {
                char buf[128];
                const char* result =
                    NcbiDumpIPRange(s_LocalIP + n, buf, sizeof(buf));
                if (result)
                    CORE_LOG_X(1, eLOG_Trace, result);
                if (s_LocalIP[n].type == eIPRange_None)
                    break;
            }
#endif /*LOCALNET_DEBUG*/
        } else
            CORE_UNLOCK;
    }
    if (ip) {
        unsigned int addr = SOCK_NetToHostLong(ip);
        for (n = 0;  n < sizeof(s_LocalIP)/sizeof(s_LocalIP[0]);  n++) {
            if (s_LocalIP[n].type == eIPRange_None)
                break;
            if (NcbiIsInIPRange(s_LocalIP + n, addr))
                return 1/*true*/;
        }
    }
    return 0/*false*/;
}


#ifdef __GNUC__
inline
#endif /*__GNUC__*/
static int/*bool*/ s_IsPrivateIP(unsigned int ip)
{
    unsigned int addr = SOCK_NetToHostLong(ip);
    return
#if defined(IN_CLASSA) && defined(IN_CLASSA_NET) && defined(IN_CLASSA_NSHIFT)
        (IN_CLASSA(addr)
         && (addr & IN_CLASSA_NET) == (IN_LOOPBACKNET << IN_CLASSA_NSHIFT))  ||
#else
        !((addr & 0xFF000000) ^ (INADDR_LOOPBACK-1))  || /* 127/8 */
#  endif /*IN_CLASSA_NET && IN_CLASSA_NSHIFT*/
        /* private [non-routable] IP ranges, according to IANA and RFC1918 */
        !((addr & 0xFF000000) ^ 0x0A000000)  || /* 10/8                      */
        !((addr & 0xFFFF0000) ^ 0xA9FE0000)  || /* 169.254/16                */
        !((addr & 0xFFF00000) ^ 0xAC100000)  || /* 172.16.0.0-172.31.255.255 */
        !((addr & 0xFFFF0000) ^ 0xC0A80000)  || /* 192.168/16                */
        /* multicast IP range is also excluded: 224.0.0.0-239.255.255.255 */
#if   defined(IN_MULTICAST)
        IN_MULTICAST(addr)
#elif defined(IN_CLASSD)
        IN_CLASSD(addr)
#else
        !((addr & 0xF0000000) ^ 0xE0000000)
#endif /*IN_MULTICAST*/
        ;
}


static const char* s_SearchTrackingEnv(const char*        name,
                                       const char* const* tracking_env)
{
    const char* result;

    if (!tracking_env) {
        result = getenv(name);
#ifdef LOCALNET_DEBUG
        CORE_LOGF(eLOG_Trace, ("Getenv('%s') = %s%s%s", name,
                               result ? "\""   : "",
                               result ? result : "NULL",
                               result ? "\""   : ""));
#endif /*LOCALNET_DEBUG*/
    } else {
        size_t len = strlen(name);
        const char* const* str;
        result = 0;
        for (str = tracking_env;  *str;  ++str) {
            if (strncasecmp(*str, name, len) == 0  &&  (*str)[len] == '=') {
                result = &(*str)[++len];
                break;
            }
        }
#ifdef LOCALNET_DEBUG
        CORE_LOGF(eLOG_Trace, ("Tracking '%s' = %s%s%s", name,
                               result ? "\""   : "",
                               result ? result : "NULL",
                               result ? "\""   : ""));
#endif /*LOCALNET_DEBUG*/
    }
    return result  &&  *(result += strspn(result, " \t")) ? result : 0;
}


static const char* s_GetForwardedFor(const char* const* tracking_env,
                                     unsigned int* addr)
{
    const char* f = s_SearchTrackingEnv("HTTP_X_FORWARDED_FOR", tracking_env);
    int/*bool*/ external;
    char *p, *q, *r, *s;
    unsigned int ip;

    if (!f)
        return 0;
    r = 0;
    external = !(s = strdup(f)) ? 1 : 0;
    for (p = s;  p  &&  *p;  p += strspn(p, ", \t")) {
        int/*bool*/ private_ip;
        q = p + strcspn(p, ", \t");
        if (*q) {
            *q++ = '\0';
        }
        if (!*p  ||  !(ip = SOCK_gethostbyname(p))) {
#ifdef LOCALNET_DEBUG
            CORE_LOG(eLOG_Trace, "Forcing external");
#endif /*LOCALNET_DEBUG*/
            external = 1;
            r        = 0;
        } else if (!(private_ip = s_IsPrivateIP(ip))  &&  !NcbiIsLocalIP(ip)) {
            r        = p;
            *addr    = ip;
            break;
        } else if (!external && (!r || (!private_ip && s_IsPrivateIP(*addr)))){
            r        = p;
            *addr    = ip;
        }
        p = q;
    }
    if (r) {
        memmove(s, r, strlen(r) + 1);
        assert(*addr);
        return s;
    }
    if (s) {
        free(s);
    }
    *addr = 0;
    return external ? "" : 0;
}


/* The environment checked here must be in correspondence with the
 * tracking environment created by CTrackingEnvHolder::CTrackingEnvHolder()
 * (header: <cgi/ncbicgi.hpp>, source: cgi/ncbicgi.cpp, library: xcgi)
 */
extern unsigned int NcbiGetCgiClientIPEx(ECgiClientIP       flag,
                                         char*              buf,
                                         size_t             buf_size,
                                         const char* const* tracking_env)
{
    struct {
        const char*  host;
        unsigned int ip;
    } probe[4];
    const char* forwarded_for = 0;
    int/*bool*/ external = 0;
    const char* host = 0;
    unsigned int ip = 0;
    size_t i;

    memset(probe, 0, sizeof(probe));
    for (i = 0;  i < sizeof(probe)/sizeof(probe[0]);  i++) {
        switch (i) {
        case 0:
            probe[i].host = s_SearchTrackingEnv("HTTP_CAF_PROXIED_HOST",
                                                tracking_env);
            break;
        case 1:
            probe[i].host = forwarded_for = s_GetForwardedFor(tracking_env,
                                                              &probe[i].ip);
            break;
        case 2:
            probe[i].host = s_SearchTrackingEnv("PROXIED_IP",
                                                tracking_env);
            break;
        case 3:
            probe[i].host = s_SearchTrackingEnv("HTTP_X_FWD_IP_ADDR",
                                                tracking_env);
            break;
        default:
            assert(0);
            continue;
        }
        if (!probe[i].host) {
            continue;
        }
        if (!probe[i].ip  &&  *probe[i].host) {
            probe[i].ip = SOCK_gethostbyname(probe[i].host);
        }
        if (*probe[i].host  &&  NcbiIsLocalIP(probe[i].ip)) {
            continue;
        }
#ifdef LOCALNET_DEBUG
        CORE_LOG(eLOG_Trace, "External on");
#endif /*LOCALNET_DEBUG*/
        external = 1;
        if (probe[i].ip  &&  !s_IsPrivateIP(probe[i].ip)) {
            assert(probe[i].host);
            host = probe[i].host;
            ip   = probe[i].ip;
            break;
        }
    }
    if (!ip) {
        for (i = external  ||  flag == eCgiClientIP_TryLeast;  i < 8;  i++) {
            unsigned int xip = 0;
            const char* xhost;
            switch (i) {
            case 0:
                assert(!external);
                xhost = s_SearchTrackingEnv("HTTP_CLIENT_HOST", tracking_env);
                break;
            case 1:
            case 2:
            case 3:
            case 4:
                xhost = probe[i - 1].host;
                xip   = probe[i - 1].ip;
                break;
            case 5:
                xhost = s_SearchTrackingEnv("REMOTE_HOST", tracking_env);
                break;
            case 6:
                xhost = s_SearchTrackingEnv("REMOTE_ADDR", tracking_env);
                break;
            case 7:
                if (flag != eCgiClientIP_TryAll) {
                    continue;
                }
                xhost = s_SearchTrackingEnv("NI_CLIENT_IPADDR", tracking_env);
                break;
            default:
                assert(0);
                continue;
            }
            if (!xhost) {
                continue;
            }
            if (i < 1  ||  4 < i) {
                xip = *xhost ? SOCK_gethostbyname(xhost) : 0;
            }
            if (!xip) {
                continue;
            }
            if (!external  ||  (!NcbiIsLocalIP(xip)  &&  !s_IsPrivateIP(xip))){
                host = xhost;
                ip   = xip;
                break;
            }
        }
    }
    assert((!ip  &&  (!host  ||  !*host))  ||
           ( ip  &&    host  &&   *host));
    if (buf  &&  buf_size) {
        if (host  &&  (i = strlen(host)) < buf_size)
            memcpy(buf, host, ++i);
        else
            buf[0] = '\0';
    }
    if (forwarded_for  &&  *forwarded_for) {
        free((void*) forwarded_for);
    }
    return ip;
}


extern unsigned int NcbiGetCgiClientIP(ECgiClientIP       flag,
                                       const char* const* tracking_env)
{
    return NcbiGetCgiClientIPEx(flag, 0, 0, tracking_env);
}


extern int/*bool*/ NcbiIsLocalCgiClient(const char* const* tracking_env)
{
    return NcbiIsLocalIP(NcbiGetCgiClientIP(eCgiClientIP_TryAll,tracking_env));
}
