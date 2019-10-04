/* $Id: ncbi_crypt_stub.c 143267 2008-10-16 18:16:07Z lavr $
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
 * Author:              Alexandra Soboleva
 * Toolkit adaptation:  Anton Lavrentiev
 *
 * File Description:
 *   Public ncbi_crypt API
 *
 */

#include <connect/ext/ncbi_crypt.h>
#include <connect/ncbi_connutil.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>


/*--------------------------------------------------------------*/
/*ARGSUSED*/
int CRYPT_Version(int version)
{
    return -1;
}


/*--------------------------------------------------------------*/
CRYPT_Key CRYPT_Init(const char* skey)
{
    return (CRYPT_Key) skey;
}


/*--------------------------------------------------------------*/
void CRYPT_Free(CRYPT_Key key)
{
    /* NOOP */
    return;
}


/*--------------------------------------------------------------*/
extern char* CRYPT_EncodeString(CRYPT_Key key, const char* str)
{
    if (key == CRYPT_BAD_KEY)
        return 0;
    return NcbiCrypt(str, (const char*) key);
}


/*--------------------------------------------------------------*/
extern char* CRYPT_DecodeString(CRYPT_Key key, const char* str)
{
    if (key == CRYPT_BAD_KEY)
        return 0;
    return NcbiDecrypt(str, (const char*) key);
}


/*--------------------------------------------------------------*/
extern char* NcbiCrypt(const char* s, const char* k)
{
    static const char kHex[] = "0123456789ABCDEF";
    size_t slen, klen, i, j;
    char *d, *pd;

    if (!s)
        return 0;
    if (!k  ||  !*k)
        return strdup(s);
    slen = strlen(s);
    if (!(d = (char*) malloc((slen << 1) + 2)))
        return 0;
    pd = d;
    *pd++ = 'H';
    klen = strlen(k);
    for (i = 0, j = 0;  i < slen;  i++, j++) {
        unsigned char c;
        if (j == klen)
            j = 0;
        c = *s++ ^ k[j];
        *pd++ = kHex[c >> 4];
        *pd++ = kHex[c & 0x0F];
    }
    *pd = '\0';
    assert(((int)(pd - d)) & 1);
    return d;
}


/*--------------------------------------------------------------*/
static unsigned char s_FromHex(char ch)
{
    unsigned char rc = ch - '0';
    if (rc <= '\x09')
        return rc;
    rc = (ch | ' ') - 'a';
    return rc <= '\x05' ? rc + '\x0A' : '\xFF';
}


extern char* NcbiDecrypt(const char* s, const char* k)
{
    size_t slen, klen, i, j;
    char *d, *pd;

    if (!s)
        return 0;
    if (!k  ||  !*k)
        return strdup(s);
    slen = strlen(s);
    if (!(slen-- & 1)  ||  *s++ != 'H')
        return 0;
    slen >>= 1;
    if (!(d = (char*) malloc(slen + 1)))
        return 0;
    pd = d;
    klen = strlen(k);
    for (i = 0, j = 0;  i < slen;  i++, j++) {
        unsigned char hi, lo;
        if ((hi = s_FromHex(*s++)) & 0xF0)
            break;
        hi <<= 4;
        if ((lo = s_FromHex(*s++)) & 0xF0)
            break;
        if (j == klen)
            j = 0;
        *pd++ = (hi | lo) ^ k[j];
    }
    *pd = '\0';
    assert(i < slen/*user error*/  ||  (size_t)(pd - d) == slen);
    return d;
}
