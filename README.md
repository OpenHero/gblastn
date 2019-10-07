#  G-BLASTN  

G-BLASTN is a GPU-accelerated nucleotide alignment tool based on the widely used NCBI-BLAST. 
G-BLASTN can produce exactly the same results as NCBI-BLAST, and it also has very similar user 
commands. It also supports a pipeline mode, which can fully utilize the GPU and CPU resources 
when handling a batch of queries. G-BLASTN supports megablast and blastn mode. The discontiguous
megablast mode is currently not supported.

Citation:

K. Zhao and X.-W. Chu, “G-BLASTN: Accelerating Nucleotide Alignment by Graphics Processors”, Oxford, Bioinformatics, 2014.
http://bioinformatics.oxfordjournals.org/content/early/2014/01/24/bioinformatics.btu047.abstract?keytype=ref&ijkey=FkuzgmzzPTJss9c 

G-BLASTN is free software and you can browse/download the source code at:
https://sourceforge.net/p/gblastn
or
https://github.com/OpenHero/gblastn


Acknowledgement:

This project is supported by grant FRG2/11-12/158 from Hong Kong Baptist University. We also thank NVIDIA corporation for their donation of GPU cards.

## News v1.2: 
1.2 [October 2019] version release is built on NCBI-BLAST 2.2.28.

Merge with ncbi-blast-2.2.28+.src support CUDA 10.1 Ubuntu 18.x GCC 7.x

### install
./configure --without-debug --with-mt --without-sybase --without-fastcgi --without-sssdb --without-sss --without-geo --without-sp --without-orbacus --without-boost
make 

### user guide
Shell g.m.sh
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


