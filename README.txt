************************ G-BLASTN 1.1 [November 2013] ****************************

G-BLASTN is a GPU-accelerated nucleotide alignment tool based on the widely used NCBI-BLAST. 
G-BLASTN can produce exactly the same results as NCBI-BLAST, and it also has very similar user 
commands. It also supports a pipeline mode, which can fully utilize the GPU and CPU resources 
when handling a batch of medium to large sized queries.

G-BLASTN is free software and you can browse/download the source code at:
https://sourceforge.net/p/gblastn
or
https://github.com/OpenHero/gblastn

News: 
1.1 version release support NCBI-BLAST 2.2.28.

I. Supported features
=====================
G-BLASTN accelerates the blastn module of NCBI-BLAST by GPUs. G-BLASTN has been tested on 
NVIDIA GPUs GTX680, GTX780, and Quadro K5000. 

Requirement:
============
1). Nvidia GPU card compute capability > 1.3
2). CUDA5.5 version. https://developer.nvidia.com/cuda-toolkit

II. Installation instructions
=============================

G-BLASTN directly modifies NCBI-BLAST 2.2.28 by adding GPU functionality. To install G-BLASTN, 
you can:


1) Download gblastn.1.1.tar.gz from https://github.com/OpenHero/gblastn/archive/gblastn.1.1.tar.gz
and unpack the package:

For example, on 64-bit Linux:

>tar zxvf gblastn.1.1.tar.gz
>cd gblastn.1.1
>chmod +x install

2) Install G-BLASTN

>./install

This will:
i.   Ask the user whether G-LBASTN should be added to an existing BLAST installation or
     whether NCBI-BLAST should be installed as well.
ii.  Modify the existing NCBI-BLAST installation or download, unpack and unzip NCBI-BLAST from
     ftp://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/2.2.28/${ncbi_blast_version}.tar.gz
     depending on what was selected by the user in (i).
iii. Compile the CUDA files.
iv.  Embed G-BLASTN into the existing or downloaded NCBI-BLAST.

If the installation is successful, you should find the executable "blastn" in
"./ncbi-blast-2.2.28+-src/c++/GCC447-ReleaseMT64/bin/".

NOTE: The directory "GCC447-ReleaseMT64" might differ on your system.

Acknowledgement: The installation configuration of G-BLASTN is based on that of GPU-BLAST 
                 (http://eudoxus.cheme.cmu.edu/gpublast/gpublast.html). 

3) The G-BLASTN 

If there is no error.
You can get the binary G-BLASTN file "blastn" in directory "/ncbi-blast-2.2.28+-src/c++/GCC447-ReleaseMT64/bin/".
Then move the "blastn" file into "bin" directy show as below(IV. Example) with command "mv".
In the "/ncbi-blast-2.2.28+-src/c++/GCC447-ReleaseMT64/bin/" directory:

>mv blastn /home/blsatn/bin/gblastn

III. How to use G-BLASTN
========================

If the above process is successful, the NCBI-BLAST installed will offer the
additional option of using G-BLASTN. The interface of G-BLASTN is identical
to the original NCBI-BLAST interface with the following additional options
for "blastn":

 *** GPU options
 -use_gpu <true|false>
   Use 'true' to enable GPU for blastn
   Default = 'false'
 -mode <1|2>
   1.normal mode, 2.pipeline mode
   Default = `1'
 -query_list <file>
   The file includes the list of files names of your query files.
 

IV. Example:
============

1) In the work directory, we can make some sub-directory:	
.
├── bin
│   ├── blastn          # the orignal blastn
│   └── gblastn         # the G-BLASTN, you can copy the binary blastn into this direcotry, and change the name to gblastn; 
├── blast
│   └── src
│       └── gpu
│           ├── c++
│           ├── install
│           ├── log
│           ├── ncbi-blast-2.2.28+-src
│           └── 2.28.zip
├── data                # nt database
│   └── nt2m
│       ├── nohup.out
│       ├── nt.2m.00.nhr
│       ├── ...
│       ├── nt.2m.counts
│       └── nt.2m.nal
├── database           # human and mouse database
│   └── blastdb
│       ├── human
│       │   ├── human.1-8.wm.body
│       │   ├── ...
│       │   ├── human.9-Y.wm.nsq
│       │   └── human.wm.nal
│       └── mouse
│           ├── mouse.1-10.wm.nhr
│           ├── ...
│           ├── mouse.11-Y.wm.nsq
│           ├── mouse.nal
│           └── mouse.wm.nal
├── output              # output directory
│   ├── query.101.fa.out
│   └── ...
├── query               # query directory
│   ├── queries
│   │   ├── human
│   │   │   ├── qlarge
│   │   │   │   ├── query.11.fa
│   │   │   │   └── ...
│   │   │   ├── qmedium
│   │   │   │   ├── query.100.fa
│   │   │   │   └── ...
│   │   │   └── qsmall
│   │   │       ├── query.101.fa
│   │   │       └── ...
│   │   └── mouse
│   │       ├── qlarge
│   │       │   ├── query.201.fa
│   │       │   └── ...
│   │       ├── qmedium
│   │       │   ├── query.101.fa
│   │       │   └── ...
│   │       └── qsmall
│   │           ├── query.100.fa
│   │           └── ...
│   └── SRR955707
│       ├── 1.fa
│       └── ...
├── script             # shell directory
│   └── new_query_human_mouse
│       ├── g.m.sh     
│       └── m.ls
└── wm_counts          # mask database for human and mouse
    ├── human.stat
    └── mouse.stat

2) The shell of g.m.sh

###################################################################################################################
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
###################################################################################################################

3) The m.ls file

Put the queris's path into this file:
/home/blastn/query/queries/mouse/qmedium/query.195.fa
/home/blastn/query/queries/mouse/qmedium/query.130.fa
