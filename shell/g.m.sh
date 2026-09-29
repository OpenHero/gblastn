#!/bin/sh
# Batch example: query-list entries are paths to individual FASTA files.
set -eu
: "${BLASTN:=blastn}"
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 database-prefix query-list output-prefix" >&2
    exit 2
fi
exec "$BLASTN" -db "$1" -query_list "$2" -out "$3" \
    -task megablast -mode 1 -evalue 1e-5 -max_target_seqs 10 -outfmt 7 -use_gpu true
