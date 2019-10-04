/* $Id: ncbi_crypt.c 182546 2010-02-01 13:49:06Z lavr $
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
 * Authors:             Oleg Khovayko     (crypt version 0)
 *                      Leonid Boitsov    (crypt version 1)
 * Toolkit adaptation:  Anton Lavrentiev  (and merging)
 *
 * File Description:
 *   Private NCBI crypting module.
 *
 *   ATTENTION!!  Not for export to third parties!!  ATTENTION!!
 *
 */

#include "../ncbi_ansi_ext.h"
#include "../ncbi_priv.h"
#include <connect/ext/ncbi_crypt.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NCBI_USE_ERRCODE_X   Connect_Crypt


#define CRYPT_DEFAULT_VERSION 1

#define CRYPT_MAGIC 0x12CC2A3
#define CRYPT_MASK  0x3F
#define CRYPT_SIZE  (CRYPT_MASK + 1)

#define CRYPT_STEP1 11
#define CRYPT_STEP2 13


#define SizeOf(a)   (sizeof(a) / sizeof(a[0]))


struct SCRYPT_KeyTag {
    int           seed;  /* place it first so we can hack it in the tests :-)*/
    short         off1;
    short         off2;
    unsigned long magic;
    const char    key[CRYPT_SIZE + 1];
};


static const    char byte_to_char[CRYPT_SIZE + 1] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789-abcdefghijklmnopqrstuvwxyz";
static unsigned char char_to_byte[256] = { 0 };


typedef void   (*FCRYPTEncoder)  (char* dst, const char* src, size_t n);
static  void   s_EncodePrintable0(char* dst, const char* src, size_t n);
static  void   s_EncodePrintable1(char* dst, const char* src, size_t n);

typedef size_t (*FCRYPTDecoder)  (char* dst, const char* src);
static  size_t s_DecodePrintable0(char* dst, const char* src);
static  size_t s_DecodePrintable1(char* dst, const char* src);

static struct SCoder {
    const char    mask;    /* crypt bitness       */
    const size_t  runlen;  /* encoding group size */
    FCRYPTEncoder encoder;
    FCRYPTDecoder decoder;
} s_Coders[] = {
    { '\x7F', 6, s_EncodePrintable0, s_DecodePrintable0 },  /* 7bit, compact */
    { '\xFF', 3, s_EncodePrintable1, s_DecodePrintable1 }   /* 8bit clean    */
};


static int s_Version = CRYPT_DEFAULT_VERSION;


/*--------------------------------------------------------------*/
int CRYPT_Version(int version)
{
    int retval = s_Version;
    if (version < 0)
        s_Version = CRYPT_DEFAULT_VERSION;
    else if (!(version & ~1))
        s_Version = version;
    return retval;
}


/*--------------------------------------------------------------*/
CRYPT_Key CRYPT_Init(const char* skey)
{
    size_t size = skey ? strlen(skey) : 0;
    CRYPT_Key key;
    char* s;

    if (!size) {
        return 0;
    }
    if (!(key = (CRYPT_Key) malloc(sizeof(*key)))) {
        return CRYPT_BAD_KEY;
    }
    if (!char_to_byte[(unsigned char) byte_to_char[CRYPT_SIZE - 1]]) {
        const char* p;
        CORE_TRACE("[CRYPT_Init]  Initializing static data");
        for (p = byte_to_char;  *p;  p++) {
            char_to_byte[(unsigned char)(*p)] = (char)((p - byte_to_char) <<2);
        }
    }
    key->seed  = (int) time(0) ^ (int) rand();
    key->off1  = key->off2 = 0;  /* delay init for later on demand */
    key->magic = CRYPT_MAGIC;
    for (s = (char*) key->key; s + size < key->key + CRYPT_SIZE; s += size) {
        memcpy(s, skey, size);
    }
    strncpy0(s, skey, (size_t)(key->key + CRYPT_SIZE - s));
    return key;
}


/*--------------------------------------------------------------*/
void CRYPT_Free(CRYPT_Key key)
{
    if (!key || key == CRYPT_BAD_KEY) {
        return;
    }
    if (key->magic != CRYPT_MAGIC) {
        CORE_LOG_X(1, eLOG_Warning, "[CRYPT_Free]  Magic corrupted");
    }
    free(key);
}


/*--------------------------------------------------------------*/
static void s_EncodePrintable0(char* dst, const char* src, size_t n)
{
    signed char lsb    = 2;
    char*       lsbptr = dst;

    while (n--) {
        char c = *src++ & 0x7F;         /* BUG: good for 7-bit chars only! */
        lsb  <<= 1;
        lsb   |= c & 1;                 /*   LSB  */
        *++dst = byte_to_char[c >> 1];  /* 6 bits */
        if (lsb < 0) {
            *lsbptr = byte_to_char[lsb & CRYPT_MASK];
            lsbptr  = ++dst;
            lsb     = 2;
        }
    }
    if (lsbptr != dst) {
        while (lsb > 0) {
            lsb <<= 1;
        }
        *lsbptr = byte_to_char[lsb & CRYPT_MASK];
        *++dst = '\0';
    } else {
        *dst = '\0';
    }
}


/*--------------------------------------------------------------*/
static size_t s_DecodePrintable0(char* dst, const char* src)
{
    char* out = dst, lsb = 0, c;
    int   bit = 1;

    while ((c = *src++) != 0) {
        c = char_to_byte[(unsigned char) c] >> 1;
        if (!--bit) {
            lsb = c;
            bit = 7;
        } else
            *out++ = c | ((lsb >> bit) & 1);
    }
    return (size_t)(out - dst);
}


/*--------------------------------------------------------------*/
static void s_EncodePrintable1(char* dst, const char* src, size_t n)
{
    signed char lsb    = 2;
    char*       lsbptr = dst;

    while (n--) {
        unsigned char c = (unsigned char)(*src++);  /* 8-bit clean version */
        lsb  <<= 2;                                
        lsb   |= c & 3;                             /* 2 LSBs */
        *++dst = byte_to_char[c >> 2];              /* 6 MSBs */
        if (lsb < 0) {
            *lsbptr = byte_to_char[lsb & CRYPT_MASK];
            lsbptr  = ++dst;
            lsb     = 2;
        }
    }
    if (lsbptr != dst) {
        while (lsb > 0) {
            lsb <<= 2;
        }
        *lsbptr = byte_to_char[lsb & CRYPT_MASK];
        *++dst = '\0';
    } else {
        *dst = '\0';
    }
}


/*--------------------------------------------------------------*/
static size_t s_DecodePrintable1(char* dst, const char* src)
{
    char* out = dst, lsb = 0, c;
    int   bit = 0;

    while ((c = *src++) != 0) {
        c = char_to_byte[(unsigned char) c];
        if (!bit) {
            lsb = c;
            bit = 6;
        } else {
            *out++ = c | ((lsb >> bit) & 3);
            bit   -= 2;
        }
    }
    return (size_t)(out - dst);
}


/*--------------------------------------------------------------*/
extern char* CRYPT_EncodeString(CRYPT_Key key, const char* str)
{
    char *out, *tmp, *t, w[1024];
    struct SCoder* coder;
    const char* src;
    int version;
    size_t len;
    char a, c;

    if (!key) {
        return str ? strdup(str) : 0;
    }
    if (key == CRYPT_BAD_KEY) {
        CORE_LOG_X(2, eLOG_Error, "[CRYPT_Encode]  Bad key");
        return 0;
    }
    if (key->magic != CRYPT_MAGIC) {
        CORE_LOG_X(3, eLOG_Error, "[CRYPT_Encode]  Bad key magic");
        return 0;
    }
    if (!str) {
        return 0;
    }
    if (key->off1 == key->off2) {
        /* Assertion:  off1 and off2 are chosen such a way that one is always
         * odd and the other one is always even, so they'll never meet
         * (stronger:  the distance between them is always an odd number,
         * let alone they are never equal, i.e. distance is never 0).
         * Here, they have not yet been initialized.
         */
        key->off1  = (short)( (key->seed       |  1) & CRYPT_MASK);   /*odd*/
        key->off2  = (short)(((key->seed >> 8) & ~1) & CRYPT_MASK);  /*even*/
    }

    len = strlen(str);
    version = s_Version;
    assert(version >= 0  &&  (size_t) version < SizeOf(s_Coders));
    coder = s_Coders + version;

    out = (char*)malloc(3 + len + (len + coder->runlen - 1)/coder->runlen + 1);
    if (!out) {
        return 0;
    }

    if (len > sizeof(w)) {
        if (!(tmp = (char*) malloc(len))) {
            free(out);
            return 0;
        }
    } else {
        tmp = w;
    }
    src = str + len;
    t = tmp;

    out[0] = '0' + (char) version;
    out[1] = byte_to_char[key->off1];
    out[2] = byte_to_char[key->off2];

    /* scramble */
    a = key->off1 + key->off2;
    while (src > str) {
        c = *--src;
        *t++ = c ^ a ^ (key->key[key->off1] + (key->key[key->off2] << 1));
        a = (a << 1) ^ (c - a);
        key->off1 += CRYPT_STEP1;
        key->off2 += CRYPT_STEP2;
        key->off1 &= CRYPT_MASK;
        key->off2 &= CRYPT_MASK;
    }

    /* make printable */
    coder->encoder(out + 3, tmp, len);

    if (tmp != w) {
        free(tmp);
    }
    return out;
}


/*--------------------------------------------------------------*/
extern char* CRYPT_DecodeString(CRYPT_Key key, const char* str)
{
    char *dst, *out, *tmp, *t, w[1024];
    struct SCoder* coder;
    short off1, off2;
    int version;
    size_t len;
    char a, c;

    if (!key) {
        return str ? strdup(str) : 0;
    }
    if (key == CRYPT_BAD_KEY) {
        CORE_LOG_X(4, eLOG_Error, "[CRYPT_Decode]  Bad key");
        return 0;
    }
    if (key->magic != CRYPT_MAGIC) {
        CORE_LOG_X(5, eLOG_Error, "[CRYPT_Decode]  Bad key magic");
        return 0;
    }
    if ((len = str ? strlen(str) : 0) < 3) {
        return 0;
    }

    version = *str++ - '0';
    if (version & ~1) {
        CORE_LOGF_X(6, eLOG_Error,
                    ("[CRYPT_Decode]  Unknown crypt version `%u'", version));
        return 0;
    }
    assert(version >= 0  &&  (size_t) version < SizeOf(s_Coders));
    coder = s_Coders + version;

    len = coder->runlen*(len - 3 + coder->runlen)/(coder->runlen + 1) + 1;

    off1 = (unsigned char) char_to_byte[(unsigned char)(*str++)] >> 2;
    off2 = (unsigned char) char_to_byte[(unsigned char)(*str++)] >> 2;

    if (len > sizeof(w)) {
        if (!(tmp = (char*) malloc(len)))
            return 0;
    } else {
        tmp = w;
    }

    /* printable string to byte values */
    len = coder->decoder(tmp, str);
    if (!(out = (char*) malloc(len + 1))) {
        if (tmp != w)
            free(tmp);
        return 0;
    }
    t = tmp;

    dst  = out + len;
    *dst = '\0';

    /* unscramble */
    a = off1 + off2;
    while (dst > out) {
        c = *t++ ^ a ^ (key->key[off1] + (key->key[off2] << 1));
        *--dst = c & coder->mask;
        a = (a << 1) ^ (c - a);
        off1 += CRYPT_STEP1;
        off2 += CRYPT_STEP2;
        off1 &= CRYPT_MASK;
        off2 &= CRYPT_MASK;
    }

    if (tmp != w) {
        free(tmp);
    }
    return out;
}


/*--------------------------------------------------------------*/
extern char* NcbiCrypt(const char* str, const char* skey)
{
    CRYPT_Key key = CRYPT_Init(skey);
    char* result = key == CRYPT_BAD_KEY ? 0 : CRYPT_EncodeString(key, str);
    CRYPT_Free(key);
    return result;
}


/*--------------------------------------------------------------*/
extern char* NcbiDecrypt(const char* str, const char* skey)
{
    CRYPT_Key key = CRYPT_Init(skey);
    char* result = key == CRYPT_BAD_KEY ? 0 : CRYPT_DecodeString(key, str);
    CRYPT_Free(key);
    return result;
}
