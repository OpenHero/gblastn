#!/bin/sh
echo "***************start*****************"
blastn="/home/kyle/testGBTN/gblastn/c++/GCC700-ReleaseMT64/bin/blastn"
data_dir="/home/kyle/ncbi/"
database="${data_dir}database/blastdb/human.wm"
maskdb="${data_dir}database/blastdb/human.stat"
query_list="m.ls"
outpath="${data_dir}log/log.txt"
 
$blastn -db $database -window_masker_db $maskdb -query_list $query_list -evalue 1e-5 -max_target_seqs 10 -outfmt "7" -out $outpath -use_gpu true

echo "***************complete*****************"