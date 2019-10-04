#ifndef CONNECT_EXT___NCBI_CRYPT__H
#define CONNECT_EXT___NCBI_CRYPT__H

/* $Id: ncbi_crypt.h 168469 2009-08-17 14:27:16Z lavr $
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
 *   Private NCBI crypting module.
 *
 *   ATTENTION!!  Not for export to third parties!!  ATTENTION!!
 *
 */

#ifdef __cplusplus
extern "C" {
#endif


/* Forward declaration of internal key handle: MT-safety */
typedef struct SCRYPT_KeyTag* CRYPT_Key;


/* Special key value to denote a failure from CRYPT_Init() */
#define CRYPT_BAD_KEY ((CRYPT_Key)(-1))


/* Build key handle based on textual "key" (not more than 64 chars taken) */
/* Return 0 if the key is empty (either 0 or "") */
extern CRYPT_Key CRYPT_Init(const char* key);


/* Free key handle (m.b. also 0 or CRYPT_BAD_KEY, both result in NOOP) */
extern void      CRYPT_Free(CRYPT_Key key);


/*
 * All [de]crypt procedures below return dynamically allocated
 * NUL-terminated character array (to be later free()'d by the caller).
 * NOTES:  key == 0 causes no (de)crypting, just a copy "strdup(str)" returned;
 *         key == CRYPT_BAD_KEY results in an error logged, 0 returned;
 *         return value 0 (w/o log) means memory allocation failure.
 */

/* Encode string "str" using key handle "key" */
extern char* CRYPT_EncodeString(CRYPT_Key key, const char* str);


/* Decode string "str" using key handle "key" */
extern char* CRYPT_DecodeString(CRYPT_Key key, const char* str);


/* COMPATIBILITY */
/* Return result of encryption of "string" with "key"; 0 if failed */
extern char* NcbiCrypt
(const char* string, const char* key);


/* COMPATIBILITY */
/* Return decryption of string with "key"; 0 if decryption failed */
extern char* NcbiDecrypt
(const char* encrypted_string, const char* key);


/* Set crypt version (to the value of the passed argument "version" >= 0),
 * or do nothing if the argument value is out of the known version limit.
 * The call is *not* recommended for use in the user's code, but solely
 * for test purposes.  The notion of default version is supported internally,
 * and "version" < 0 causes the default version to become effective.
 * Return the actual version that has been acting prior to the call, or
 * -1 if the version may not be changed (e.g. stub variant of the library).
 */
extern int CRYPT_Version(int version);


#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CONNECT_EXT___NCBI_CRYPT__H */
