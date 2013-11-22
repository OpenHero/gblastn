#include <algo/blast/gpu_blast/utility.h>
#include <string.h>
#include <fstream>
#include <iostream>
#include <sstream>

#if 0
RecordsMap::RecordsMap(char* fa_table_file_path) {
	char line[255];
	
	std::string buf;
	int start_offset = 0;
	int end_offset = 0;
	char * pch;

	std::vector<Record*>* records = NULL;
	Record* record = NULL;
	FILE* fr;

	fr = fopen(fa_table_file_path, "r");
	if ( fr == NULL ) {
		printf("Unable to open the file : %s\n", fa_table_file_path);
		exit(0);
	}

	while ( fgets(line, 255, fr) != NULL ) {
		std::vector<std::string> tokens;
		pch = strtok (line,"|,");
		int i = 0;
		while ( pch != NULL ) {
			tokens.push_back(std::string(pch));
			pch = strtok (NULL, "|,");
			i++;
		}

		start_offset = atoi(tokens[i-2].c_str());
		end_offset = atoi(tokens[i-1].c_str());
		
		record = new Record(tokens[0], tokens[1], tokens[2], 
			tokens[3], tokens[4].append(",").append(tokens[5]), 
			start_offset, end_offset);

		if ( start_offset == 0 ) {
			records = new std::vector<Record*>();
			records->push_back(record);

			_records_map[tokens[1]] = records;
		}
		else {
			records->push_back(record);
		}
	}

	fclose(fr);
}

RecordsMap::RecordsMap(string fa_table_file_path) {
	
	std::string buf;
	int start_offset = 0;
	int end_offset = 0;
	char * pch;

	std::vector<Record*>* records = NULL;
	Record* record = NULL;
	fstream fr;
	fr.open(fa_table_file_path);
	if ( !fr.is_open()) {
		cout <<"Unable to open the file : " << fa_table_file_path <<endl;
		exit(0);
	}

	string line;
	while ( getline(fr, line)) {
		
		string id = line.substr(1,line.find_first_of(" ")-2);
		line = line.substr(line.find_last_of("|"));
		start_offset = atoi(line.substr(1,line.find_first_of(",")-1).c_str());
		end_offset = atoi(line.substr(line.find_first_of(",")+1).c_str());

		/*record = new Record(tokens[0], tokens[1], tokens[2], 
			tokens[3], tokens[4].append(",").append(tokens[5]), 
			start_offset, end_offset);

		if ( start_offset == 0 ) {
			records = new std::vector<Record*>();
			records->push_back(record);

			_records_map[tokens[1]] = records;
		}
		else {
			records->push_back(record);
		}*/
	}

	fr.close();
}

RecordsMap::~RecordsMap() {
	for ( std::map<std::string, std::vector<Record*>*>::iterator it = _records_map.begin(); it != _records_map.end(); ++it ) {
		// Remove all elements in the list
		std::vector<Record*>* records = (std::vector<Record*>*)(it->second);
		if ( records != NULL ) {
			for ( int i = 0; i < records->size(); i++ ) {
				delete records->at(i);
			}
			delete records;
		}		
	}
	_records_map.clear();
}

std::vector<Record*>* RecordsMap::getRecords(std::string key) {
	return _records_map[key];
}

Record* RecordsMap::getCorrectedRecord(std::string key, int& start_offset, int& end_offset) {
	std::vector<Record*>* records = _records_map[key];
	if ( records != NULL ) {
		// End offset larger than the last one
		int size = records->size();
		if ( start_offset < records->at(0)->get_start_offset() || records->at(size-1)->get_end_offset() < end_offset )
			return NULL;

		int low = 0;
		int high = size-1;
		int mid = 0;
		int offset = 0;

		while ( low <= high ) {
			mid = (low + high)/2;
			offset = records->at(mid)->get_start_offset();
			if ( offset == start_offset ) {
				break;
			}
			else if ( offset < start_offset ) {
				low = mid + 1;
			}
			else if ( offset > start_offset ) {
				high = mid - 1;
			}
		}

		mid = (low + high)/2;
		Record* record = NULL;
		if ( records->at(mid)->locate_within_range(start_offset, end_offset) ) {
			record = records->at(mid);
			start_offset -= record->get_start_offset();
			end_offset -= record->get_start_offset();
		}
		else if ( size >= 2 && records->at(mid)->locate_within_range(start_offset, end_offset) ) {
			record = records->at(mid+1);
		}
		else {
			record = NULL;
		}

		return record;

		/*
		for (int i = 0; i < records->size(); i++ ) {
			if ( records->at(i).locate_within_range(start_offset, end_offset) ) {
				Record* record = &records->at(i);
				start_offset -= record->get_start_offset();
				end_offset -= record->get_start_offset();
				return record;
			}
		}
		*/
	}
	else 
		return NULL;
}
#endif
#if 0

Record::Record(std::string source, std::string gi_number, std::string gb, std::string accession, std::string locus, int start_offset, int end_offset) {
	_source = source;
	_gi_number = gi_number;
	_gb = gb;
	_accession = accession;
	_locus = locus;

	_start_offset = start_offset;
	_end_offset = end_offset;
}

bool Record::locate_within_range(int start_offset, int end_offset) {
	return (_start_offset <= start_offset && end_offset <= _end_offset);
}

int Record::get_start_offset() {
	return _start_offset;
}

int Record::get_end_offset() {
	return _end_offset;
}

std::string Record::get_gi_number() {
	return _gi_number;
}

std::string Record::get_accession() {
	return _accession;
}

std::string Record::get_shorten_locus() {
	char locus[255];
	strcpy(locus, _locus.c_str());
	std::vector<std::string> tokens;
	char* pch = strtok(locus," ");
	while ( pch != NULL ) {
		tokens.push_back(std::string(pch));
		pch = strtok (NULL, " ");
	}

	return std::string(tokens[0]);
}

std::string Record::get_locus() {
	return _locus;
}
#endif

//////////////////////////////////////////////////////////////////////////
//
NewRecord::NewRecord(string id, unsigned int start_offset, unsigned int end_offset)
{
	_id = id;
	_start_offset = start_offset;
	_end_offset = end_offset;
}

bool NewRecord::locate_within_range(int start_offset, int end_offset) {
	return (_start_offset <= start_offset && end_offset <= _end_offset);
}


NewRecordsMap::NewRecordsMap(string fa_table_file_path) {
	
	std::string buf;
	unsigned int start_offset = 0;
	unsigned int end_offset = 0;
	char * pch;

	vector<NewRecord*>* records = NULL;
	NewRecord* record = NULL;
	fstream fr;
	fr.open(fa_table_file_path.c_str());
	if ( !fr.is_open()) {
		cout <<"Unable to open the file : " << fa_table_file_path <<endl;
		exit(0);
	}

	string line;
	while ( getline(fr, line)) {
		
		string id = line.substr(1,line.find_first_of(" ")-1);
		line = line.substr(line.find_last_of("|"));
		stringstream myStrStart(line.substr(1,line.find_first_of(",")-1).c_str());
		stringstream myStrEnd(line.substr(line.find_first_of(",")+1).c_str());

		myStrStart >>start_offset;
		myStrEnd >> end_offset;

		record = new NewRecord(id, start_offset, end_offset);

		if ( start_offset == 0 ) {
			records = new std::vector<NewRecord*>();
			records->push_back(record);

			_records_map[id] = records;
		}
		else {
			records->push_back(record);
		}
	}

	fr.close();
}

NewRecordsMap::~NewRecordsMap() {
	for ( map<string, vector<NewRecord*>*>::iterator it = _records_map.begin(); it != _records_map.end(); ++it ) {
		// Remove all elements in the list
		std::vector<Record*>* records = (std::vector<Record*>*)(it->second);
		if ( records != NULL ) {
			for ( int i = 0; i < records->size(); i++ ) {
				delete records->at(i);
			}
			delete records;
		}		
	}
	_records_map.clear();
}


NewRecord* NewRecordsMap::getCorrectedRecord(string key, unsigned int& start_offset, unsigned int& end_offset) {
	vector<NewRecord*>* records = _records_map[key];
	if ( records != NULL ) {
		// End offset larger than the last one
		int size = records->size();
		if ( start_offset < records->at(0)->_start_offset || records->at(size-1)->_end_offset < end_offset )
			return NULL;

		int low = 0;
		int high = size-1;
		int mid = 0;
		int offset = 0;

		while ( low <= high ) {
			mid = (low + high)/2;
			offset = records->at(mid)->_start_offset;
			if ( offset == start_offset ) {
				break;
			}
			else if ( offset < start_offset ) {
				low = mid + 1;
			}
			else if ( offset > start_offset ) {
				high = mid - 1;
			}
		}

		mid = (low + high)/2;
		NewRecord* record = NULL;
		if ( records->at(mid)->locate_within_range(start_offset, end_offset) ) {
			record = records->at(mid);
			start_offset -= record->_start_offset;
			end_offset -= record->_start_offset;
		}
		else if ( size >= 2 && records->at(mid)->locate_within_range(start_offset, end_offset) ) {
			record = records->at(mid+1);
		}
		else {
			record = NULL;
		}

		return record;
	}
	else 
		return NULL;
}
