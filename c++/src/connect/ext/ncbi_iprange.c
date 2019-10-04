/* $Id: ncbi_iprange.c 360937 2012-04-27 13:47:28Z lavr $
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
 *   IP range manipulating API
 *
 */

#include "../ncbi_ansi_ext.h"
#include <connect/ext/ncbi_iprange.h>
#include <connect/ncbi_socket.h>
#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>


int/*bool*/ NcbiIsInIPRange(const SIPRange* range, unsigned int addr)
{
    if (range  &&  addr) {
        switch (range->type) {
        case eIPRange_Host:
            assert(range->a == range->b  ||  !range->b);
            return !(addr ^ range->a);
        case eIPRange_Range:
            assert(range->a <= range->b);
            return range->a <= addr  &&  addr <= range->b;
        case eIPRange_Network:
            assert(range->a  &&  range->b);
            return !((addr & range->b) ^ range->a);
        default:
            assert(0);
            /*FALLTHRU*/
        case eIPRange_None:
            break;
        }
    }
    return 0/*false*/;
}


SIPRange NcbiTrueIPRange(const SIPRange* range)
{
    SIPRange retval;
    if (range) {
        switch (range->type) {
        case eIPRange_Host:
            retval.a =  range->a;
            retval.b =  range->a;
            break;
        case eIPRange_Range:
            retval.a =  range->a;
            retval.b =  range->b;
            break;
        case eIPRange_Network:
            retval.a =  range->a;
            retval.b = (range->a & range->b) | ~range->b;
            break;
        default:
            assert(0);
            /*FALLTHRU*/
        case eIPRange_None:
            memset(&retval, 0, sizeof(retval));
            /*retval.type = eIPRange_Range;*/
            return retval;
        }
        retval.type = eIPRange_Range;
    } else
        memset(&retval, 0, sizeof(retval));
    return retval;
}


const char* NcbiDumpIPRange(const SIPRange* range, char* buf, size_t bufsize)
{
    char result[128];
    if (!range  ||  !buf  ||  !bufsize)
        return 0;
    if (range->type != eIPRange_None) {
        char* s = result;
        SIPRange temp = NcbiTrueIPRange(range);
        switch (range->type) {
        case eIPRange_Host:
            strcpy(s, "Host");
            s += 4;
            break;
        case eIPRange_Range:
            strcpy(s, "Range");
            s += 5;
            break;
        case eIPRange_Network:
            strcpy(s, "Network");
            s += 7;
            break;
        default:
            assert(0);
            return 0;
        }
        *s++ = ' ';
        if (SOCK_ntoa(SOCK_HostToNetLong(temp.a),
                      s, sizeof(result) - (size_t)(s - result)) != 0) {
            strcpy(s++, "?");
        } else
            s += strlen(s);
        *s++ = '-';
        if (SOCK_ntoa(SOCK_HostToNetLong(temp.b),
                      s, sizeof(result) - (size_t)(s - result)) != 0) {
            strcpy(s, "?");
        }
    } else
        strcpy(result, "None");
    return strncpy0(buf, result, bufsize - 1);
}


int/*bool*/ NcbiParseIPRange(SIPRange* range, const char* s)
{
    if (!range  ||  !s)
        return 0/*false*/;
    if (!*s) {
        memset(range, 0, sizeof(*range));
        /*range->type = eIPRange_None;*/
        return 1/*success*/;
    }
    if (!SOCK_isip(s)) {
        const char* p = s;
        int dots = 0;
        range->type = eIPRange_Host;
        for (;;) {
            char small[4];
            char* e;
            long d;

            if (*p != '*') {
                errno = 0;
                d = strtol(p, &e, 10);
                if (errno  ||  p == e  ||  e - p > 3  ||  d < 0  ||  d > 255)
                    break/*goto out*/;
                sprintf(small, "%u", (unsigned int) d);
                if (strlen(small) != (size_t)(e - p))
                    break/*goto out*/;
                p = e;
            } else if (!*++p  &&  dots) {
                unsigned int shift = (4 - dots) << 3;
                range->type = eIPRange_Range;
                range->a  <<= shift;
                range->b    = range->a;
                range->b   |= (1 << shift) - 1;
                return 1/*true*/;
            } else
                return 0/*false*/;
            switch (range->type) {
            case eIPRange_Host:
                range->a <<= 8;
                range->a  |= d;
                if (*p != '.') {
                    range->a <<= (3 - dots) << 3;
                    switch (*p) {
                    case '/':
                        range->type = eIPRange_Network;
                        break;
                    case '-':
                        range->type = eIPRange_Range;
                        p++;
                        continue;
                    default:
                        goto out;
                    }
                } else if (++dots <= 3) {
                    p++;
                    continue;
                } else
                    goto out;
                assert(*p == '/'  &&  range->type == eIPRange_Network);
                if (!SOCK_isipEx(++p, 1/*fullquad*/))
                    continue;
                range->b  = SOCK_NetToHostLong(SOCK_gethostbyname(p));
                return range->a  &&  !(range->a & ~range->b)
                    &&  ~range->b  &&  !(~range->b & (~range->b + 1));
            case eIPRange_Range:
                if (*p)
                    goto out;
                range->b  = dots > 0 ? range->a : 0;
                range->b &= ~((1 << ((4 - dots) << 3)) - 1);
                range->b |=    d << ((3 - dots) << 3);
                range->b |=   (1 << ((3 - dots) << 3)) - 1;
                if (range->a == range->b)
                    range->type = eIPRange_Host;
                return range->a <= range->b;
            case eIPRange_Network:
                if (*p  ||  d > 32)
                    return 0/*failure (slashes are not allowed in hostnames)*/;
                if (!d  ||  d == 32) {
                    range->type = eIPRange_Host;
                    return 1/*success*/;
                }
                range->b = ~((1 << (32 - d)) - 1);
                return range->a  &&  !(range->a & ~range->b);
            default:
                assert(0);
                return 0/*failure*/;
            }
        }
    out:
        /* last resort (and maybe expensive one): try as a regular host name */
        ;
    } else { /* NB: SOCK_gethostbyname() returns 0 on an unknown host */
        size_t n;
        for (n = 0;  n < 4;  n++) {
            size_t len = 1 + (n << 1);
            if (strncmp(s, "0.0.0.0", len) == 0  &&  !s[len]) {
                range->type = eIPRange_Host;
                range->a    = 0;
                range->b    = 0;
                return 1/*success*/;
            }
        }
        /* forbid misleading IP representations */
        if (!SOCK_isipEx(s, 1/*fullquad*/))
            return 0/*failure*/;
    }
    if (!(range->a = SOCK_gethostbyname(s)))
        return 0/*failure*/;
    range->type = eIPRange_Host;
    range->a    = SOCK_NetToHostLong(range->a);
    range->b    = 0;
    return 1/*success*/;
}
