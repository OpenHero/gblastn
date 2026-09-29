# Compatibility and validation record

Work started from clean commit `1f931d5910150f44e8ceab81599428027703c879` on
2026-09-29. No unrelated checkout changes were present. Algorithm baseline:
G-BLASTN / NCBI BLAST **2.2.28+**.

## Observed target

| Component | Observed value |
|---|---|
| GPU | NVIDIA Thor, compute capability 11.0, integrated/shared memory |
| Host architecture | Linux aarch64 |
| Host OS | Ubuntu 24.04.4 LTS |
| Kernel | 6.8.12-1021-tegra |
| Jetson Linux | 39.2.1, package `39.2.1-20260806224157` |
| JetPack release corresponding to BSP | 7.2.1; JetPack meta-package not installed/reported by dpkg |
| Host driver | 595.78; CUDA driver API reports 13020 |
| NVIDIA Container Toolkit | 1.19.1-1 |
| Container OS | Ubuntu 24.04.4 LTS |
| CUDA image | NVIDIA CUDA 13.2.1, Ubuntu 24.04, linux/arm64 |
| nvcc | 13.2.78, `cuda_13.2.r13.2/compiler.37668154_0` |
| Host compiler in builder | GCC/G++ 13.3.0, Ubuntu package `13.3.0-6ubuntu2~24.04.1` |
| Code generation | Native sm_110 plus compute_110 PTX |
| Container engine used | Rootless Podman; local Ubuntu crun 1.14.1 with retained GPU groups |

The pinned development index is
`sha256:44a9504c6dfb50b1241464241b02a93871928f373de6f5a644cf5fe9f080aa63`;
its ARM64 manifest is
`sha256:0e1392f431f89f143d0d6d0fa397a2b9a6a236f8b3628cfd3afbf21e15ab4a98`.
The pinned runtime index is
`sha256:4c17c2e7474ca1d470428b25e87e95756f51e7da3b663859894a03bc5a7ebdae`;
its ARM64 manifest is
`sha256:a52783d8d73ace53998d4e740515e9942e73072dc7fbd5322917eb382a0bc7fb`.
Raw manifest metadata is retained in [`images.lock.json`](../container/images.lock.json).

## Checks completed

- Python fixture/oracle unit tests: pass. They reject missing, duplicated,
  numerically different, incorrectly dispatched and non-kernel evidence.
- Mocked CUDA host lifecycle test: pass with macOS Clang. Proves lazy discovery,
  CPU/no-device behavior, GPU-ID lease lookup and cleanup, not CUDA execution.
- Scalar execution of the actual kernel helper bodies: pass under host
  AddressSanitizer and UndefinedBehaviorSanitizer, including signed Int2 lookup
  chains, bounded tails, context boundaries and scan ordering.
- Native aarch64 CUDA 13.2.1 compilation of the GPU translation unit: pass,
  with no GPU passed through to compilation.
- Native aarch64 GCC13 compilation of GPU preliminary-engine/controller sources:
  pass.
- Standalone `tests/device_probe.cu`: actual sm_110 kernel completes on Thor,
  both on the host and in the pinned CUDA development container. This is an
  environment probe and is **not** evidence that BLAST scan/extension kernels ran.
- `tests/cuda_kernel_checks.cu`: actual production CUDA scan/mini-extension bodies
  pass six configurations on Thor (small lookup widths/strides and MB widths 11/12,
  including repetitive chains and unaligned starts/tails) against independent
  basewise references. Compute Sanitizer reports **0 memcheck errors, 0 racecheck
  hazards, and 0 synccheck errors**. Raw evidence:
  [`validation/thor-cuda-kernel-checks.log`](../validation/thor-cuda-kernel-checks.log).

End-to-end development-container checks also passed on Thor:

| Suite | Result |
|---|---|
| No-device CPU/database/CLI suite | Pass; GPU request fails explicitly |
| CPU/GPU alignment and pipeline comparisons | Pass; both lookup families execute real scan/extension kernels |
| Full Compute Sanitizer memcheck suite | Pass; 26 instrumented searches, 0 errors |
| Full Compute Sanitizer racecheck suite | Pass; 26 instrumented searches, 0 errors or warnings |
| Full Compute Sanitizer synccheck suite | Pass; 26 instrumented searches, 0 errors |
| CPU pipeline stress | Pass; 160 runs: modes 0/2 × 1/2 threads × 40 repetitions |
| CUDA header incremental rebuild | Pass; touching an included `.cuh` recompiles CUDA object and archive |
| Executable architecture inspection | `cuobjdump` confirms sm_110 cubin plus embedded PTX |

The pipeline stress test exposed and verified a fix for shared sequence-source
ownership: preliminary search and traceback had reused a mutable `copied` flag,
allowing an mmap pointer to be released with `free()`. Each search now owns an
independent sequence-source wrapper. Independent per-batch options also prevent
temporary effective-search-space changes from racing traceback.

The ABI probe on Thor observed native `BlastContextInfo` size 32, alignment 8,
query field offsets 0/4 on both host and device. The actually transferred wire
layout is size 8, alignment 4, offsets 0/4, and the device round-trip verified
transferred first/last context fields. Thus correctness does not depend on the
historical `--no-align-double` workaround.

## Packaged images

A clean native ARM64 build produced the binaries installed in both final images.
The last cached packaging rebuild updated only tests. Image configuration,
executable hashes, versions, resolved libraries and CUDA object inspection are
retained in [`thor-image-checks.json`](../validation/thor-image-checks.json).

| Image on Thor | Image ID |
|---|---|
| `localhost/gblastn:thor` | `d2e3a5ec749f912494e4590feafab3aabf27b6cc6ccbfc972d8ec37075f7c992` |
| `localhost/gblastn:thor-validation` | `1de52c11cd87fb045f99de3f3fec6d8bcebd7a8bbb65bbb54f0d08c4666cfa33` |

Both contain identical `blastn`, `makeblastdb` and `blastdbcmd` binaries.
The `blastn` SHA256 is
`6a4e2b0245a54b49605ddab3c800219450903a4a59f40d56af56d5204c60a332`.
All runtime dependencies resolve, and version output contains only the expected
BLAST version lines. Removing the base image's banner entrypoint also preserves
tabular search stdout. The portable run wrapper produced files owned by the
invoking UID/GID (1000:1000), and its stdout matched file-directed CPU output.

The clean build and final packaging logs are retained under [`validation/`](../validation/).

## Final-image acceptance

All suites below ran on the final image IDs above through the portable Podman
wrapper. Expected-failure commands are checked for nonzero exits and the relevant
diagnostic; they are included in command counts.

| Suite | Result and retained command record |
|---|---|
| No GPU passthrough | Pass, [56 commands](../validation/final-no-device/results.json); CPU/database/help/version work, strict GPU request fails |
| Thor GPU acceptance | Pass, [82 commands](../validation/final-gpu/results.json); exact CPU/GPU alignment equality, thresholds, both kernel families, pipelines and failure paths |
| Compute Sanitizer memcheck | Pass, [82 commands](../validation/release-memcheck/results.json), including 26 instrumented searches; 0 errors |
| Compute Sanitizer racecheck | Pass, [82 commands](../validation/release-racecheck/results.json), including 26 instrumented searches; 0 hazards, errors or warnings |
| Compute Sanitizer synccheck | Pass, [82 commands](../validation/release-synccheck/results.json), including 26 instrumented searches; 0 errors |
| Final runtime CPU pipeline stress | Pass, [160 runs](../validation/release/cpu-pipeline-stress/results.json); modes 0/2 × threads 1/2 × 40 repetitions, exact equality and all three batches completed |

[`final-evidence.json`](../validation/final-evidence.json) records outer wrapper
commands, image IDs, per-search sanitizer summaries and log hashes. Representative
raw BLAST output, dispatch/kernel/ABI/memory traces and sanitizer output are saved
alongside the results; [`validation/README.md`](../validation/README.md) maps them.
The four local Python/oracle/mock-lifecycle tests, shell syntax checks and
`git diff --check` also pass. The local mock tests do not substitute for the Thor
results above.

Representative measurements from the final unsanitized GPU suite, with both
strands, DUST disabled and one search thread:

| Fixture | CPU wall time | GPU wall time | Tracked GPU allocation peak |
|---|---:|---:|---:|
| Small lookup, four subjects around 500,000 bases | 0.0160 s | 0.3010 s | 308,569 bytes |
| MB lookup, four subjects around 10,000 bases | 0.0251 s | 0.2413 s | 16,937,622 bytes |

These are single observations including process/CUDA startup, not a throughput
benchmark. GPU runs were slower on these small correctness fixtures. The tracked
allocation counter excludes driver/context memory and Thrust temporary storage.
Per-command JSON also records CPU user/system time; its `children_maxrss` field
is the **cumulative child-process high-water mark**, not per-command or GPU memory.
No speedup is claimed.

## Scope of the result

The equality oracle is this repository's preserved 2.2.28+ CPU path. An
independent separately built NCBI 2.2.28+ executable was not run; the harness
supports one through `--reference`. No comparison with a newer BLAST baseline
is presented as equivalence evidence. Standalone production-kernel tests also
compare against independent basewise references.

Container runtime validation used rootless Podman with crun and CDI. Docker's
daemon was not exercised on this host. Discontiguous megablast GPU execution,
general `eNaLookupTable` GPU searches, converted-database remapping, multiple
pipeline stage workers, loaded megablast indexes with GPU execution and query
lists combined with imported search strategies are explicitly unsupported.
No multi-GPU scaling or large-dataset throughput claim is made.

## Environment findings

The first local ARM64 VM build failed while pulling CUDA because the macOS host
filesystem ran out of space. Compilation and hardware work moved to the isolated
directory `/tmp/gblastn-thor-QB8xuZNb` on the existing Thor host.

On Thor, Docker's socket was unavailable to the account. Rootless Podman could
build, but its default `runc` lost supplementary GPU groups even with
`--group-add keep-groups`. CUDA discovery failed with `/dev/nvmap` access errors
although `nvidia-smi` listed the GPU. Running with a user-local copy of Ubuntu's
`crun`, existing CDI `nvidia.com/gpu=all`, and retained supplementary groups fixed
the probe. Host driver, firmware, Docker configuration and device permissions were
not modified; no privileged container was required.

See [`THOR.md`](THOR.md) for exact build, run and acceptance commands. GPU-free
CPU tests, actual CUDA acceptance, and Compute Sanitizer checks have separate
result directories and must not be conflated.
