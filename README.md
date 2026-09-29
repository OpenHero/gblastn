#  G-BLASTN  

G-BLASTN is a GPU-accelerated nucleotide alignment tool based on the widely used NCBI-BLAST. 
This repository retains the **NCBI BLAST 2.2.28+** algorithmic baseline and real CUDA
scan/mini-extension kernels. It supports `blastn`, contiguous `megablast`, and bounded
query pipelines. Discontiguous megablast GPU execution is unsupported.

The Thor port targets Linux ARM64, Ubuntu 24.04, CUDA 13.x and **sm_110**. See the
[build, container and validation guide](docs/THOR.md) for exact commands, CPU/GPU
behavior, v4 database compatibility, pipeline modes and limitations, and the
[compatibility and test record](docs/VALIDATION.md) for measured validation status.

```sh
export CONTAINER_ENGINE=podman
export OCI_RUNTIME=crun # A user-local executable path also works.
JOBS=8 ./scripts/build-container.sh
GPU=0 ./scripts/run-container.sh blastn -version
```

Unset `CONTAINER_ENGINE` to use Docker instead.

The multi-stage image includes matching `blastn`, `makeblastdb`, and `blastdbcmd`.
CUDA base image digests and ARM64 manifests are recorded in
[container/images.lock.json](container/images.lock.json). Build success alone is
not GPU correctness evidence; the acceptance suite requires completed search kernels.

Citation:

K. Zhao and X.-W. Chu, “G-BLASTN: Accelerating Nucleotide Alignment by Graphics Processors”, Oxford, Bioinformatics, 2014.
http://bioinformatics.oxfordjournals.org/content/early/2014/01/24/bioinformatics.btu047.abstract?keytype=ref&ijkey=FkuzgmzzPTJss9c 

@article{KY.Zhao2014,  
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

Merged with ncbi-blast-2.2.28+.src support CUDA 10.1 Ubuntu 18.x GCC 7.x

The original 2019 CUDA 10.1/GCC 7 installation instructions have been superseded by
[scripts/build.sh](scripts/build.sh) and the [Thor guide](docs/THOR.md).
Portable examples are in [shell/](shell/).


## More details
https://www.comp.hkbu.edu.hk/~chxw/software/G-BLASTN.html

## License
See the [LICENSE](LICENSE.md) file for license rights and limitations (MIT).
