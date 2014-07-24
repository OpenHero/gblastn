************************ G-BLASTN 1.1 [November 2013] ****************************

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

News: 
1.1 version release is built on NCBI-BLAST 2.2.28.

I. Supported features
=====================
G-BLASTN accelerates the blastn module of NCBI-BLAST by GPUs. G-BLASTN has been tested on 
NVIDIA GPUs GTX680, GTX780, and Quadro K5000. 

Requirement:
============
1). Nvidia GPU card with compute capability > 1.3 <br \>
2). CUDA5.5 version. https://developer.nvidia.com/cuda-toolkit

II. Installation instructions
=============================

https://github.com/OpenHero/gblastn/wiki/Install-guide

III. How to use G-BLASTN
========================
https://github.com/OpenHero/gblastn/wiki/Use-guide
