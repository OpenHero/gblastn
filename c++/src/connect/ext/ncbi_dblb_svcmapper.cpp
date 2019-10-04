/* $Id: ncbi_dblb_svcmapper.cpp 340179 2011-10-05 20:23:39Z ivanovp $
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

#include <ncbi_pch.hpp>

#include <connect/ext/ncbi_dblb_svcmapper.hpp>
#include <connect/ext/ncbi_dblb.h>
#include <connect/ncbi_socket.hpp>
#include <connect/ncbi_service.h>
#include <corelib/ncbiapp.hpp>
#include <algorithm>


BEGIN_NCBI_SCOPE


///////////////////////////////////////////////////////////////////////////////
class CCharInserter
{
public:
    CCharInserter(vector<const char*>& names)
    : m_Names(&names)
    {
    }

public:
    void operator()(const TSvrRef& server)
    {
        m_Names->push_back(server->GetName().c_str());
    }

private:
    vector<const char*>* m_Names;
};



///////////////////////////////////////////////////////////////////////////////
CDBLB_ServiceMapper::CDBLB_ServiceMapper(const IRegistry* registry)
{
    ConfigureFromRegistry(registry);
}


CDBLB_ServiceMapper::~CDBLB_ServiceMapper(void)
{
}


void
CDBLB_ServiceMapper::Configure(const IRegistry* registry)
{
    CFastMutexGuard mg(m_Mtx);

    ConfigureFromRegistry(registry);
}


void
CDBLB_ServiceMapper::ConfigureFromRegistry(const IRegistry* registry)
{
    // Get current registry ...
    if (!registry && CNcbiApplication::Instance()) {
        registry = &CNcbiApplication::Instance()->GetConfig();
    }
    if (registry)
        m_EmptyTTL = registry->GetInt("dblb", "cached_empty_service_ttl", 1);
    else
        m_EmptyTTL = 1;
}


TSvrRef
CDBLB_ServiceMapper::GetServer(const string& service)
{
    CFastMutexGuard mg(m_Mtx);

    TSrvSet& exclude_list = m_ExcludeMap[service];

    time_t cur_time = time(NULL);
    ERASE_ITERATE(TSrvSet, it, exclude_list) {
        if ((*it)->GetExpireTime() <= cur_time) {
            _TRACE("For " << service << ": erasing from excluded server '"
                   << (*it)->GetName() << "', host " << (*it)->GetHost()
                   << ", port " << (*it)->GetPort());
            exclude_list.erase(it);
            m_LBEmptyMap.erase(service);
        }
    }

    TLBEmptyMap::iterator it = m_LBEmptyMap.find(service);
    if (it != m_LBEmptyMap.end()) {
        if (it->second >= cur_time) {
            // We've tried this service already. It is not served by load
            // balancer. There is no reason to try it again.
            _TRACE("Service " << service << " is known dead, bypassing LBSM.");
            return TSvrRef();
        }
        else {
            m_LBEmptyMap.erase(it);
        }
    }

    vector<const char*> skip_names;
    std::for_each(exclude_list.begin(),
                  exclude_list.end(),
                  CCharInserter(skip_names));
    skip_names.push_back(NULL);

    SDBLB_Preference preference;
    TSvrRef preferred_svr = m_PreferenceMap[service].second;
    if (!preferred_svr.Empty()) {
        preference.host = preferred_svr->GetHost();
        preference.port = preferred_svr->GetPort();
        preference.pref = m_PreferenceMap[service].first;
    }

    SDBLB_ConnPoint cp;
    char name_buff[256];
    EDBLB_Status status;

    const char* svr_name = ::DBLB_GetServer(service.c_str(),
                                            fDBLB_AllowFallbackToStandby,
                                            preferred_svr.Empty()
                                                            ? 0
                                                            : &preference,
                                            &skip_names.front(),
                                            &cp,
                                            name_buff,
                                            sizeof(name_buff),
                                            &status);

    if (cp.time == 0) {
        cp.time = TNCBI_Time(cur_time) + 10;
    }

    if (svr_name  &&  *svr_name  &&  status != eDBLB_NoDNSEntry) {
        return TSvrRef(new CDBServer(svr_name,  cp.host, cp.port, cp.time));
    } else if (cp.host) {
        return TSvrRef(new CDBServer(kEmptyStr, cp.host, cp.port, cp.time));
    }

    _TRACE("Remembering: service " << service << " is dead.");
    m_LBEmptyMap[service] = cur_time + m_EmptyTTL;
    return TSvrRef();
}

void
CDBLB_ServiceMapper::GetServersList(const string& service, list<string>* serv_list) const
{
    serv_list->clear();
    SConnNetInfo* net_info = ConnNetInfo_Create(service.c_str());
    SERV_ITER srv_it = SERV_Open(service.c_str(),
                                 fSERV_Standalone | fSERV_IncludeDown,
                                 0, net_info);
    ConnNetInfo_Destroy(net_info);
    const SSERV_Info* sinfo;
    while ((sinfo = SERV_GetNextInfo(srv_it)) != NULL) {
        if (sinfo->time > 0  &&  sinfo->time != NCBI_TIME_INFINITE) {
            string server_name(CSocketAPI::ntoa(sinfo->host));
            if (sinfo->port != 0) {
                server_name.append(1, ':');
                server_name.append(NStr::UIntToString(sinfo->port));
            }
            serv_list->push_back(server_name);
        }
    }
    SERV_Close(srv_it);
}

void
CDBLB_ServiceMapper::Exclude(const string&  service,
                             const TSvrRef& server)
{
    CFastMutexGuard mg(m_Mtx);

    _TRACE("For " << service << ": excluding server '" << server->GetName()
           << "', host " << server->GetHost()
           << ", port " << server->GetPort());
    m_ExcludeMap[service].insert(server);
}


void
CDBLB_ServiceMapper::CleanExcluded(const string& service)
{
    CFastMutexGuard mg(m_Mtx);

    _TRACE("For " << service << ": cleaning excluded list");
    m_ExcludeMap[service].clear();
}


void
CDBLB_ServiceMapper::SetPreference(const string&  service,
                                   const TSvrRef& preferred_server,
                                   double         preference)
{
    CFastMutexGuard mg(m_Mtx);

    m_PreferenceMap[service] = make_pair(preference, preferred_server);
}


IDBServiceMapper*
CDBLB_ServiceMapper::Factory(const IRegistry* registry)
{
    return new CDBLB_ServiceMapper(registry);
}

///////////////////////////////////////////////////////////////////////////////
string
CDBServiceMapperTraits<CDBLB_ServiceMapper>::GetName(void)
{
    return "DBLB_NAME_MAPPER";
}


END_NCBI_SCOPE
