#!/bin/sh
echo "***************start*****************"
blastn="/home/blastn/bin/gblastn"
data_dir="/home/blastn/"
database="${data_dir}database/blastdb/mouse/mouse.wm"
maskdb="${data_dir}wm_counts/mouse.stat"
query_list="m.ls"
outpath="/home/blastn/output/log.txt"
 
$blastn -db $database -window_masker_db $maskdb -query_list $query_list -outfmt "7" -out $outpath -use_gpu true

echo "***************complete*****************"