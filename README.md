#  G-BLASTN  

G-BLASTN is a GPU-accelerated nucleotide alignment tool based on the widely used NCBI-BLAST. 
G-BLASTN can produce exactly the same results as NCBI-BLAST, and it also has very similar user 
commands. It also supports a pipeline mode, which can fully utilize the GPU and CPU resources 
when handling a batch of queries. G-BLASTN supports megablast and blastn mode. The discontiguous
megablast mode is currently not supported.

Citation:

K. Zhao and X.-W. Chu, “G-BLASTN: Accelerating Nucleotide Alignment by Graphics Processors”, Oxford, Bioinformatics, 2014.
http://bioinformatics.oxfordjournals.org/content/early/2014/01/24/bioinformatics.btu047.abstract?keytype=ref&ijkey=FkuzgmzzPTJss9c 

@article{KY Zhao2014,  
author = {Kaiyong Zhao and Xiaowen Chu},  
title = {{G-BLASTN: accelerating nucleotide alignment by graphics processors}},  
journal = {Bioinformatics},  
year = {2014},  
volume = {30},  
number = {10},  
pages = {1384-1391},  
doi = {10.1093/bioinformatics/btu047},  
url = {https://academic.oup.com/bioinformatics/article/30/10/1384/267507}  
}  

Acknowledgement:

This project is supported by grant FRG2/11-12/158 from Hong Kong Baptist University. We also thank NVIDIA corporation for their donation of GPU cards.

## News v1.2: 
1.2 [October 2019] version release is built on NCBI-BLAST 2.2.28.

Merge with ncbi-blast-2.2.28+.src support CUDA 10.1 Ubuntu 18.x GCC 7.x

### Install
./configure --without-debug --with-mt --without-sybase --without-fastcgi --without-sssdb --without-sss --without-geo --without-sp --without-orbacus --without-boost

make 

### User guide
Please follow the example in shell directory.


## More details
https://www.comp.hkbu.edu.hk/~chxw/software/G-BLASTN.html