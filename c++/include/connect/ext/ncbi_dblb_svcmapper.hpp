#ifndef CONNECT_EXT___NCBI_DBLB_SVCMAPPER__HPP
#define CONNECT_EXT___NCBI_DBLB_SVCMAPPER__HPP

/* $Id: ncbi_dblb_svcmapper.hpp 341365 2011-10-19 14:10:03Z lavr $
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
 * Author:  Sergey Sikorskiy
 *
 * File Description:
 *   Database service name to server name mapping policy implementation.
 *
 */

#include <corelib/ncbimtx.hpp>
#include <corelib/impl/ncbi_dbsvcmapper.hpp>
#include <connect/ncbi_core_cxx.hpp>
#include <set>
#include <map>


BEGIN_NCBI_SCOPE


///////////////////////////////////////////////////////////////////////////////
/// CDBLBServerNamePolicy
///

class CDBLB_ServiceMapper : public    IDBServiceMapper,
                            protected CConnIniter
                            
{
public:
    CDBLB_ServiceMapper(const IRegistry* registry = NULL);
    virtual ~CDBLB_ServiceMapper(void);

    virtual void    Configure    (const IRegistry* registry = NULL);
    virtual TSvrRef GetServer    (const string&    service);
    virtual void    Exclude      (const string&    service,
                                  const TSvrRef&   server);
    virtual void    CleanExcluded(const string&    service);
    virtual void    GetServersList(const string& service, list<string>* serv_list) const;
    virtual void    SetPreference(const string&    service,
                                  const TSvrRef&   preferred_server,
                                  double           preference = 100.0);

    static IDBServiceMapper* Factory(const IRegistry* registry);

protected:
    void ConfigureFromRegistry(const IRegistry* registry = NULL);

private:
    typedef set<TSvrRef, SDereferenceLess>       TSrvSet;
    typedef map<string, TSrvSet>                 TExcludeMap;
    typedef map<string, time_t>                  TLBEmptyMap;
    typedef map<string, pair<double, TSvrRef> >  TPreferenceMap;

    CFastMutex      m_Mtx;
    TExcludeMap     m_ExcludeMap;
    TLBEmptyMap     m_LBEmptyMap;
    TPreferenceMap  m_PreferenceMap;
    int             m_EmptyTTL;
};


///////////////////////////////////////////////////////////////////////////////
template <>
class CDBServiceMapperTraits<CDBLB_ServiceMapper>
{
public:
    static string GetName(void);
};


END_NCBI_SCOPE

#endif  /* CONNECT_EXT___NCBI_DBLB_SVCMAPPER__HPP */
