# Retained Thor validation evidence

These records were generated on the actual NVIDIA Thor host during the
2026-09-29 port. Full compatibility details and interpretation are in
[`docs/VALIDATION.md`](../docs/VALIDATION.md). Logs contain expected failures as
well as successful commands; each suite's `results.json` records overall status.

- `thor-clean-build.log`: native ARM64 clean compilation.
- `thor-runtime-packaging.log`, `thor-validation-packaging.log`: final packaging.
- `thor-image-checks.json`: final image IDs, executable/test hashes, tool versions,
  resolved runtime libraries and sm_110 cubin/PTX inspection.
- `thor-cuda-incremental.log`: touching a CUDA header rebuilds the object/archive.
- `thor-cuda-kernel-checks.log`: independent-reference production-kernel checks
  and all three Compute Sanitizer tools.
- `final-evidence.json`: final wrapper commands, image IDs, measurements and all
  26 clean sanitizer summaries per tool, with raw-output hashes.
- `final-no-device/`: packaged runtime without GPU passthrough, including the
  explicit missing-device error and full command/timing record.
- `final-gpu/`: final runtime CPU/GPU acceptance, representative matching TSVs and
  raw completed-kernel, threshold, ABI and memory traces for both lookup families.
- `release-memcheck/`, `release-racecheck/`, `release-synccheck/`: final validation
  image acceptance records and representative raw sanitizer output.
- `release/cpu-pipeline-stress/`: 160 successful final-runtime stress comparisons,
  full commands, hashes and representative three-batch traces.
- `development/`: earlier development-container acceptance records, retained
  separately from final-image results.

Complete generated fixtures, outputs and per-command logs remain on
`valthos-remote` under `/tmp/gblastn-thor-QB8xuZNb/validation/`. Only representative
raw outputs and compact command records are copied here. The temporary host
directory is not needed to reproduce the tests: use the commands in
[`docs/THOR.md`](../docs/THOR.md).

The no-device and GPU suites contain 56 and 82 commands respectively. Each full
sanitizer suite contains 82 commands, of which 26 GPU searches run under the
selected sanitizer. These numbers include checked failure cases; they are not
counts of independent biological datasets.
