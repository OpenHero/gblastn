************************ G-BLASTN 1.0 [August 2013] ****************************

G-BLASTN is a GPU-accelerated nucleotide alignment tool based on the widely used NCBI-BLAST. 
G-BLASTN can produce exactly the same results as NCBI-BLAST, and it also has very similar user 
commands. It also supports a pipeline mode, which can fully utilize the GPU and CPU resources 
when handling a batch of medium to large sized queries.

G-BLASTN is free software.

I. Supported features
=====================
G-BLASTN accelerates the blastn module of NCBI-BLAST by GPUs. G-BLASTN has been tested on 
NVIDIA GPUs GTX680, GTX780, and Quadro K5000. 



II. Installation instructions
=============================

G-BLASTN directly modifies NCBI-BLAST 2.2.26 by adding GPU functionality. To install G-BLASTN, 
you can:

1) Download gpu_blastn_linux_configure.tar.gz from http://sourceforge.net/projects/gblastn/
and unpack the package:

For example, on 64-bit Linux:

>tar zxvf gpu_blastn_linux_configure.tar.gz
>./chmod +x install

2) Install G-BLASTN

>./install ncbi-blast-2.2.26+-src

This will:
i.   Ask the user whether G-LBASTN should be added to an existing BLAST installation or
     whether NCBI-BLAST should be installed as well.
ii.  Modify the existing NCBI-BLAST installation or download, unpack and unzip NCBI-BLAST from
     ftp://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/2.2.26
     depending on what was selected by the user in (i).
iii. Compile the CUDA files.
iv.  Embed G-BLASTN into the existing or downloaded NCBI-BLAST.

If the installation is successful, you should find the executable "blastn" in
"./ncbi-blast-2.2.26+-src/c++/GCC447-ReleaseMT64/bin/".

NOTE: The directory "GCC447-ReleaseMT64" might differ on your system.

Acknowledgement: The installation configuration of G-BLASTN is based on that of GPU-BLAST 
                 (http://eudoxus.cheme.cmu.edu/gpublast/gpublast.html). 



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
 