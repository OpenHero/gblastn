#ifndef _H_UTILITY_H_
#define _H_UTILITY_H_

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <hash_map>
#include <map>
#include <string>

using namespace std;

class Record {
	private:
		string _source;
		string _gi_number;
		string _gb;
		string _accession;
		string _locus;
		int _start_offset;
		int _end_offset;
	public:
		Record(string source, string gi_number, string gb, string accession, string locus, int start_offset, int end_offset);
		bool locate_within_range(int start_offset, int end_offset);
		int get_start_offset();
		int get_end_offset();
		string get_gi_number();
		string get_accession();
		string get_locus();
		string get_shorten_locus();
};

struct NewRecord
{
public:
	NewRecord(string id, unsigned int start_offset, unsigned int end_offset);
	bool locate_within_range(int start_offset, int end_offset);

	string _id;
	unsigned int _start_offset;
	unsigned int _end_offset;
};

#if 1
class RecordsMap {
	private:
		std::map<string, std::vector<Record*>*> _records_map;
	public:
		RecordsMap(char* fa_table_file_path);
		RecordsMap(string fa_table_file_path);
		~RecordsMap();
		std::vector<Record*>* getRecords(string key);
		Record* getCorrectedRecord(string key, int& start_offset, int& end_offset);
};
#endif

class NewRecordsMap
{
private:
	map<string, vector<NewRecord*>*> _records_map;
public:
	NewRecordsMap(string fa_table_file_path);
	~NewRecordsMap();
	NewRecord* getCorrectedRecord(string key, unsigned int& start_offset, unsigned int& end_offset);
};


#endif //_H_UTILITY_H_