#ifndef CONNECT_EXT___NCBI_IPRANGE__H
#define CONNECT_EXT___NCBI_IPRANGE__H

/* $Id: ncbi_iprange.h 267969 2011-03-28 13:44:09Z lavr $
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

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    eIPRange_None = 0,
    eIPRange_Host,
    eIPRange_Range,
    eIPRange_Network
} EIPRangeType;


typedef struct {
    EIPRangeType type;
    unsigned int a, b;  /* host byte order */
} SIPRange;


int/*bool*/ NcbiIsInIPRange(const SIPRange* range,
                            unsigned int    addr/*host byte order*/);

SIPRange    NcbiTrueIPRange(const SIPRange* range);


const char* NcbiDumpIPRange(const SIPRange* range, char* buf, size_t bufsize);

#define     NcbiPrintIPRange(b, bs, r)  NcbiDumpIPRange(r, b, bs)


int/*bool*/ NcbiParseIPRange(SIPRange* range, const char* s);


#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /*CONNECT_EXT___NCBI_IPRANGE__H*/
