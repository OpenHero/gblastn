# Jetson Thor build and validation

This is a platform port of G-BLASTN's **NCBI BLAST 2.2.28+** baseline. It retains
CUDA nucleotide scanning and mini-extension; BLAST's scoring, diagonal filtering,
gapped extension, masking and formatting remain on the existing baseline.
Contiguous `megablast` and `blastn` are supported search tasks. Discontiguous
megablast GPU execution is rejected and has not been validated.

## Platform and image selection

On 2026-09-29, [NVIDIA's JetPack release page](https://developer.nvidia.com/embedded/jetpack/downloads)
lists JetPack 7.2.1 / Jetson Linux 39.2.1 / CUDA 13.2.2. Thor is ARM64, compute
capability **11.0**, and the build generates **sm_110 machine code and compute_110
PTX**. See NVIDIA's [Thor CUDA guide](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/0.1.0/setup_cuda.html).
No GPU or host NVIDIA driver is needed during compilation.

NVIDIA's registry did not publish a `13.2.2-devel-ubuntu24.04` tag when checked.
The Dockerfile pins the published `nvidia/cuda:13.2.1-devel-ubuntu24.04` and matching
runtime image by immutable multi-platform index digest. Both contain verified
`linux/arm64` manifests. Exact index and ARM64 member digests are in
[`container/images.lock.json`](../container/images.lock.json). A toolkit patch
version in JetPack does not imply an identically named container exists.

The host supplies the driver; the image supplies CUDA runtime libraries. Do not
install a server driver in the container or replace the Jetson BSP driver.
Check the actual host before choosing an image:

```sh
uname -m
cat /etc/nv_tegra_release
cat /etc/os-release
dpkg-query -W nvidia-l4t-core nvidia-jetpack nvidia-container-toolkit
nvidia-smi
nvidia-ctk cdi list
```

Follow the [Thor container setup guide](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/0.1.0/setup_docker.html)
to have an administrator configure Docker and NVIDIA Container Toolkit if needed.
The repository scripts do not change host Docker configuration, drivers or firmware.
Docker uses `--gpus all --runtime=nvidia`; rootless Podman can use the existing
CDI device `--device=nvidia.com/gpu=all`. Neither requires privileged containers.

## Build

Build natively on Thor when available. A build in another ARM64 Linux environment
establishes compilation only, not Thor execution compatibility.

For the validated rootless Podman setup, select the engine once for all following
build and run examples:

```sh
export CONTAINER_ENGINE=podman
export OCI_RUNTIME=crun # Or the absolute path to a user-local crun executable.
```

The scripts default to Docker when `CONTAINER_ENGINE` is unset.

```sh
JOBS=8 ./scripts/build-container.sh
# Retain compiler, cuobjdump and Compute Sanitizer for validation:
JOBS=8 TARGET=validation IMAGE=gblastn:thor-validation ./scripts/build-container.sh
# Explicit clean build:
JOBS=8 ./scripts/build-container.sh --no-cache
```

`CONTAINER_ENGINE=podman` selects Podman. Rootless GPU runs require a runtime that
preserves the host's GPU groups: use `OCI_RUNTIME=crun` (or an absolute path to
`crun`); the wrapper supplies `--group-add keep-groups` and `--userns=keep-id`.
On the validation host, `runc` dropped those supplementary groups and CUDA
discovery failed, although `nvidia-smi` could still list the device. A user-local
extracted Ubuntu `crun` package fixed this without changing the host configuration.
For example:

```sh
mkdir -p work
CONTAINER_ENGINE=podman OCI_RUNTIME=crun WORK_DIR="$PWD/work" \
  ./scripts/run-container.sh blastn -help
```

`JOBS`, `CUDA_ARCH` (default 110), `IMAGE`
and `TARGET` are configurable. For another supported host release, verify and
override **both** base image digests with `--build-arg CUDA_DEVEL_IMAGE=...` and
`--build-arg CUDA_RUNTIME_IMAGE=...`. Do not use an image newer than the host's
supported CUDA stack without independently establishing compatibility.

Inside an appropriate Linux CUDA development environment, the existing NCBI
configure/make build is used:

```sh
CC=gcc-13 CXX=g++-13 CUDA_HOME=/usr/local/cuda CUDA_ARCH=110 \
  PREFIX="$PWD/install" BUILD_ROOT="$PWD/.build/toolkit" JOBS=8 ./scripts/build.sh
```

`NVCC`, `CUDA_LIBDIR`, `CFLAGS`, `CXXFLAGS`, `CPPFLAGS` and `LDFLAGS` can also be
provided. The legacy host toolkit uses GNU C++14; the CUDA/Thrust translation unit
uses C++17. GCC 13 is supported by the selected CUDA toolkit. Native toolkit
libraries are linked statically where the original build supports this, while
the CUDA runtime is deliberately provided by the runtime image. It is not copied
manually from a developer workstation. Packaged executables are `blastn`,
`makeblastdb`, and `blastdbcmd`; `ldd` checks run during packaging.

## Databases and searches

The included `makeblastdb` writes **BLAST database v4**, which this baseline reads.
Newer tools defaulting to v5 and arbitrary downloaded databases are not assumed
compatible. Keep the database sidecar files together in the mounted directory.

```sh
mkdir -p work
python3 tests/fixtures.py "$PWD/work/fixtures"
WORK_DIR="$PWD/work" GPU=0 ./scripts/run-container.sh makeblastdb \
  -in fixtures/small_subjects.fa -dbtype nucl -parse_seqids -out fixtures/small.db
WORK_DIR="$PWD/work" GPU=0 ./scripts/run-container.sh blastdbcmd -db fixtures/small.db -info

WORK_DIR="$PWD/work" GPU=0 ./scripts/run-container.sh blastn \
  -task blastn -db fixtures/small.db -query fixtures/small.fa \
  -mode 1 -use_gpu false -outfmt 6 -out cpu.tsv
WORK_DIR="$PWD/work" ./scripts/run-container.sh blastn \
  -task blastn -db fixtures/small.db -query fixtures/small.fa \
  -mode 1 -use_gpu true -gpu_required true -outfmt 6 -out gpu.tsv
```

The run script maps the invoking user's UID/GID, so output remains user-owned.
To use separate directories with Docker directly:

```sh
docker run --rm --gpus all --runtime=nvidia --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -v "$PWD/queries:/queries:ro" -v "$PWD/databases:/db:ro" \
  -v "$PWD/results:/results" gblastn:thor blastn \
  -task megablast -db /db/reference -query /queries/input.fa \
  -use_gpu true -gpu_required true -mode 1 -outfmt 6 -out /results/hits.tsv
```

GPU discovery happens only when a search requests CUDA. Help, version, database
utilities and `-use_gpu false` operate without GPU passthrough. GPU requests fail
on unavailable devices, allocation failure or CUDA errors; there is no silent
CPU fallback for a failed CUDA request. `-gpu_required true` additionally rejects
contradictory `-use_gpu false` invocations. It does **not** override the original
per-subject dispatch thresholds. A GPU-selected search can legitimately use CPU
callbacks on short subjects, so strict validation also requires kernel evidence.

Device selection, compute capability, backend and query status go to stderr.
`GBLASTN_GPU_TRACE=1` adds per-subject dispatch, completed scan/mini-extension
kernel counts, layout verification and tracked device-allocation high-water marks.
Completion is reported after checked device-to-host copies, not merely after a
launch. Normal execution does not add a device-wide synchronization per kernel.
`GBLASTN_GPU_MEMORY_LIMIT_MB` defaults to 1024 and accepts 16–16384; it limits
tracked application device allocations on Thor's shared memory. Driver/context
and Thrust internal temporary storage are additional memory consumers.

## Query pipelines

`-mode` now matches dispatch, validates values, joins all workers and propagates
errors:

| Value | Behavior |
|---|---|
| `0` | Pipeline preliminary search and traceback within each query file; drain between files. |
| `1` | Sequential query processing. Default. |
| `2` | Pipeline across files in `-query_list`, retaining output order. |
| `3` | Compatibility alias for `2` (the old help documented this value). |

Pipelines retain one preliminary stage and one traceback stage, bounding live
query batches. Use `-num_threads` for search threading. Legacy stage-count options
greater than one are rejected. Converted-database coordinate remapping is
unsupported; use ordinary v4 nucleotide databases. Imported saved strategies
work for single-query execution; combining them with `-query_list` is rejected.
GPU searches reject loaded megablast indexes that bypass CUDA callbacks; use
`-use_index false` for that case.

A query list contains one path per line, interpreted relative to the process's
working directory **inside the container**. Blank lines are skipped. `-out` selects
the output directory through its parent path; each file becomes `basename.fa.out`.
Missing/empty lists, missing/empty inputs and basename collisions fail before
batch dispatch. An individual query failure makes the process fail; earlier
successful output files may remain and are not a complete batch.

```sh
mkdir -p work/pipeline
printf '%s\n' fixtures/small.fa fixtures/megablast.fa > work/queries.list
WORK_DIR="$PWD/work" ./scripts/run-container.sh blastn \
  -task blastn -db fixtures/small.db -query_list queries.list \
  -mode 2 -use_gpu true -gpu_required true -outfmt 6 -out pipeline/batch.out
```

## Acceptance

The tests generate deterministic FASTA locally, build v4 databases with the
matching executable, and compare CPU/GPU identifiers, coordinates, strand frames,
identity, gaps, raw and bit scores, and E-values. Only row order is normalized;
duplicate multiplicity and numeric differences fail. Newer NCBI releases are not
the equality oracle. `--reference /path/to/blastn` optionally adds an independent
2.2.28+ CPU executable and verifies its version first.

Small-NA lookup subjects are 499999, 500000, 500001 and 500003 bases. MB lookup
subjects are 9999, 10000, 10001 and 10003 bases. Only values **strictly greater**
than 500000 / 10000 select the respective GPU callbacks. MB query padding ensures
the actual MB lookup is chosen: merely specifying `-task megablast` with a tiny
query can select the small lookup. Tests require both kernel phases for both
branches and verify the observed per-subject dispatch. Coverage includes exact,
substitution, gap, ambiguous, masked, reverse-strand, tail, no-hit and multi-subject
cases, repeated runs, pipeline modes and failing batches.

```sh
python3 -m unittest discover -s tests -p 'test_*.py' -v
mkdir -p validation
# No GPU devices passed; explicit CPU succeeds and strict CUDA must fail:
WORK_DIR="$PWD/validation" GPU=0 ./scripts/run-container.sh \
  python3 /opt/gblastn/tests/acceptance.py --suite no-device --work /work/no-device
# Thor, with real CUDA kernel evidence:
WORK_DIR="$PWD/validation" ./scripts/run-container.sh \
  python3 /opt/gblastn/tests/acceptance.py --suite gpu --work /work/gpu
```

Compute Sanitizer runs in the development-based validation image:

```sh
for tool in memcheck racecheck synccheck; do
  WORK_DIR="$PWD/validation" IMAGE=gblastn:thor-validation ./scripts/run-container.sh \
    python3 /opt/gblastn/tests/acceptance.py --suite gpu --sanitizer "$tool" \
    --timeout 1800 --work "/work/$tool"
done
```

For fast isolation of CUDA arithmetic/scheduling problems, the production kernel
headers also have standalone host and real-device checks:

```sh
c++ -std=c++17 -fsanitize=address,undefined tests/kernel_host_checks.cpp -o /tmp/kernel-host
/tmp/kernel-host
# Inside the CUDA development/validation container, from /src/gblastn:
nvcc -std=c++17 -arch=sm_110 -lineinfo tests/cuda_kernel_checks.cu -o /tmp/kernel-cuda
/tmp/kernel-cuda
for tool in memcheck racecheck synccheck; do
  compute-sanitizer --tool "$tool" --error-exitcode 86 /tmp/kernel-cuda
done
```

These small checks supplement full BLAST acceptance. `BATCH_SIZE=500` in the full
suite forces multiple batches per query file; submission/completion traces prove
that these batches finish in each pipeline mode. Word sizes 7, 9 and 12 cover
additional lookup/direct-extension paths. The standard default-word cases still
require both completed CUDA scan and mini-extension phases.

Sanitizer findings produce nonzero exit status. Logs, full commands, wall/CPU
timings, CPU memory high-water marks and a `results.json` pass/fail record remain
in each work directory. `GBLASTN_MEMORY` stderr records report tracked GPU
allocation use. These are measurements, not a promised speedup. Sanitizers alter
timings substantially. Hardware and validation results are recorded separately
in [`docs/VALIDATION.md`](VALIDATION.md).

## Porting details and limits

Legacy global texture references are replaced by query-owned device pointers,
including signed `Int2` small-table entries. Bounded count/prefix/emit scanning
keeps subject and lookup-chain order and avoids shared append-buffer overflow.
Per-input mini-extension results preserve stable hit order before the existing
scored extension. Packed sequence tails use bounded byte reads. Allocation failure
and a single chain exceeding capacity are errors, rather than dropped hits.

`BlastContextInfo` is no longer copied as an implicit host/device ABI. The two
device-consumed fields use an explicit 8-byte, 4-byte-aligned wire structure with
offset assertions, plus a device layout/field round-trip probe. Native structure
layout diagnostics retain visibility into the reason for the old
`--no-align-double` workaround. ARM64 reference-count and pointer operations use
compiler atomics with explicit ordering. Architecture-specific SSE selection is
guarded; no unconditional `-mssse3` is passed on ARM64. Existing aarch64 support in
`config.guess`/`config.sub` is retained.

No multi-GPU throughput or speedup claim is made. GPU selection does not imply all
work runs on CUDA: scoring/traceback and below-threshold subjects retain the
original CPU work. Acceptance on the actual supported host remains the release
gate.

The GPU controller supports the small-NA and contiguous MB lookup families.
Options/query sizes that select the baseline's general `eNaLookupTable` are
rejected for GPU execution; use explicit CPU execution for that lookup family.
