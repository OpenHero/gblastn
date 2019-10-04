#ifndef CGI___CAF_ENCODED__HPP
#define CGI___CAF_ENCODED__HPP

/*  $Id: caf_encoded.hpp 103491 2007-05-04 17:18:18Z kazimird $
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
* Author:  Anatoliy Kuznetsov
*
* File Description:
*   Encoded implementation of cookie affinity interface
*
*/

#include "caf_plain.hpp"
#include "connect/ext/ncbi_crypt.h"
#include "internal/webenv2/id.h"


/** @addtogroup CookieAffinity
 *
 * @{
 */


BEGIN_NCBI_SCOPE

///////////////////////////////////////////////////////
//
// CCookieAffinity_Encoded::
//
// Cookie affinity service interface implementation with
// NCBI specific web encoding
//

class NCBI_XCGI_EXPORT CCookieAffinity_Encoded : public CCookieAffinity_Plain
{
public:
    // Return result of encryption of "string" with "key"; 0 if failed
    // return string to be free()'d by the caller
    // Function always uses NCBI specific encoding key
    virtual char* Encode(const char* str, const char* key);
};


/* @} */


///////////////////////////////////////////////////////
//  CCookieAffinity_Encoded::
//

inline char* CCookieAffinity_Encoded::Encode(const char* str,
                                             const char* /*key*/)
{
    return NcbiCrypt(str, WEBENV_CRYPTKEY);  // always use WEBENV_CRYPTKEY
}


END_NCBI_SCOPE

#endif  /* CGI___CAF_ENCODED__HPP */
