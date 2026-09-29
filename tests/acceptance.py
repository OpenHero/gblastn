#!/usr/bin/env python3
"""CPU/no-device checks and strict Thor acceptance, against the same executable."""
import argparse
import collections
import decimal
import json
import os
import platform
import re
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

from fixtures import generate

FIELDS = "qseqid sseqid qstart qend sstart send pident nident length mismatch gaps gapopen score bitscore evalue qframe sframe"


def rows(path):
    """Only row ordering is ignored; duplicates and every numeric field matter."""
    result = collections.Counter()
    for line in Path(path).read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        values = line.split("\t")
        if len(values) != len(FIELDS.split()):
            raise AssertionError(f"Unexpected BLAST output in {path}: {line}")
        result[tuple(values[:2]) + tuple(decimal.Decimal(v) for v in values[2:])] += 1
    return result


def compare(left, right):
    expected, actual = rows(left), rows(right)
    if expected != actual:
        raise AssertionError(f"Alignment mismatch {left} vs {right}\n"
                             f"Missing: {expected - actual}\nExtra: {actual - expected}")


class Acceptance:
    def __init__(self, args):
        self.args = args
        self.root = args.work.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.data = self.root / "fixtures"
        self.manifest = generate(self.data)
        self.records = []
        self.env = dict(os.environ, GBLASTN_GPU_TRACE="1", GBLASTN_ABI_CHECK="1")
        # Lookup selection depends on batch size. Keep main branch-evidence
        # tests independent of an inherited shell experiment; exercise the
        # explicit override separately in multiple_batches().
        self.env.pop("BATCH_SIZE", None)

    def run(self, label, command, fail=False, sanitize=False):
        if sanitize:
            command = ["compute-sanitizer", "--tool", self.args.sanitizer,
                       "--error-exitcode", "86", "--target-processes", "all"] + command
        start = time.monotonic()
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        proc = subprocess.run(command, env=self.env, cwd=self.root, capture_output=True,
                              text=True, timeout=self.args.timeout)
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        (self.root / f"{label}.stdout").write_text(proc.stdout)
        (self.root / f"{label}.stderr").write_text(proc.stderr)
        self.records.append({"label": label, "command": command, "returncode": proc.returncode,
                             "wall_seconds": time.monotonic() - start,
                             "user_seconds": after.ru_utime - before.ru_utime,
                             "system_seconds": after.ru_stime - before.ru_stime,
                             "children_maxrss": after.ru_maxrss,
                             "maxrss_units": "bytes" if sys.platform == "darwin" else "KiB"})
        print(f"{label}: exit={proc.returncode} {self.records[-1]['wall_seconds']:.3f}s", flush=True)
        if (proc.returncode == 0) == fail:
            raise AssertionError(f"Unexpected exit from {label}: {proc.returncode}\n{proc.stderr[-6000:]}")
        if sanitize:
            summary = ("RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)"
                       if self.args.sanitizer == "racecheck" else "ERROR SUMMARY: 0 errors")
            if summary not in proc.stdout + proc.stderr:
                raise AssertionError(f"Missing clean Compute Sanitizer summary for {label}")
        return proc

    def command(self, branch="small", gpu=False, mode=1, strand="both", dust="no"):
        return [self.args.blastn, "-task", "blastn" if branch == "small" else "megablast",
                "-db", str(self.data / f"{branch}.db"), "-query", str(self.data / f"{branch}.fa"),
                "-outfmt", "6 " + FIELDS, "-evalue", "1e-10", "-num_threads", "1",
                "-strand", strand, "-dust", dust, "-soft_masking", "true", "-lcase_masking",
                "-mode", str(mode), "-use_gpu", str(gpu).lower()] + (["-gpu_required", "true"] if gpu else [])

    def prepare(self):
        self.run("help", [self.args.blastn, "-help"])
        self.run("version", [self.args.blastn, "-version"])
        self.run("makeblastdb-version", [self.args.makeblastdb, "-version"])
        for branch in self.manifest["datasets"]:
            self.run(f"make-{branch}", [self.args.makeblastdb, "-in", str(self.data / f"{branch}_subjects.fa"),
                     "-dbtype", "nucl", "-parse_seqids", "-out", str(self.data / f"{branch}.db")])
            self.run(f"db-info-{branch}", [self.args.blastdbcmd, "-db", str(self.data / f"{branch}.db"), "-info"])
            # 2.2.28 makeblastdb writes v4. Verify the actual on-disk header.
            header = (self.data / f"{branch}.db.nin").read_bytes()[:4]
            if int.from_bytes(header, "big") != 4:
                raise AssertionError("Acceptance requires an actual format v4 database")

    @staticmethod
    def evidence(stderr, branch, phases=("scan", "extend")):
        if not re.search(r'GBLASTN_DEVICE backend=cuda id=\d+ name="[^"]*Thor[^"]*" compute_capability=11\.0\b', stderr):
            raise AssertionError("GPU acceptance requires an actual Thor device (compute capability 11.0)")
        for phase in phases:
            pattern = rf"GBLASTN_KERNEL branch={branch} phase={phase} status=completed\b"
            if not re.search(pattern, stderr):
                raise AssertionError(f"Missing completed {branch}/{phase} CUDA kernel evidence")
        threshold = 500000 if branch == "small" else 10000
        for length in (threshold - 1, threshold, threshold + 1, threshold + 3):
            backend = "cuda" if length > threshold else "cpu"
            pattern = rf"GBLASTN_DISPATCH branch={branch} subject_length={length} backend={backend}\b"
            if not re.search(pattern, stderr):
                raise AssertionError(f"Missing dispatch evidence: {branch}, length {length}, {backend}")

    def searches(self):
        gpu = self.args.suite == "gpu"
        for branch in ("small", "megablast"):
            for strand, dust in (("both", "no"), ("plus", "yes"), ("minus", "no")):
                label = f"{branch}-{strand}-dust-{dust}"
                cpu = self.root / f"{label}-cpu.tsv"
                self.run(label + "-cpu", self.command(branch, strand=strand, dust=dust) + ["-out", str(cpu)])
                if strand == "both":
                    observed = rows(cpu)
                    ids = {row[0] for row in observed}
                    if not {"exact", "substitutions", "gap", "ambiguous", "reverse", "tail"} <= ids or "nohit" in ids:
                        raise AssertionError(f"Fixture did not exercise expected hit/no-hit cases: {ids}")
                    if not any(row[0] == "gap" and row[10] > 0 for row in observed):
                        raise AssertionError("Gap fixture produced no gapped alignment")
                    if not any(row[0] == "substitutions" and row[9] > 0 for row in observed):
                        raise AssertionError("Substitution fixture produced no mismatches")
                    if not any(row[0] == "reverse" and row[4] > row[5] for row in observed):
                        raise AssertionError("Reverse-strand fixture produced no reverse alignment")
                    for length in self.manifest["datasets"][branch]["subject_lengths"]:
                        if not any(row[0] == "tail" and row[1].endswith(str(length)) and max(row[4], row[5]) == length
                                   for row in observed):
                            raise AssertionError(f"Tail fixture did not align through final base of length {length}")
                if gpu:
                    for repeat in range(2):
                        out = self.root / f"{label}-gpu-{repeat}.tsv"
                        proc = self.run(f"{label}-gpu-{repeat}", self.command(branch, True, strand=strand, dust=dust)
                                        + ["-out", str(out)], sanitize=bool(self.args.sanitizer))
                        self.evidence(proc.stderr, branch)
                        compare(cpu, out)
                if self.args.reference:
                    command = self.command(branch, strand=strand, dust=dust)
                    # Independent 2.2.28+ executable has no G-BLASTN-specific options.
                    command = [self.args.reference] + command[1:command.index("-mode")]
                    out = self.root / f"{label}-reference.tsv"
                    self.run(label + "-reference", command + ["-out", str(out)])
                    compare(cpu, out)

    def pipelines(self):
        for mode in (0, 1, 2, 3):
            outputs = []
            for gpu in ([False, True] if self.args.suite == "gpu" else [False]):
                label = f"pipeline-{mode}-{'gpu' if gpu else 'cpu'}"
                directory = self.root / label
                directory.mkdir(exist_ok=True)
                command = self.command(gpu=gpu, mode=mode)
                # Keep the large-subject database but use word_size=28 so both
                # small and MB lookups need real GPU mini-extension in a batch.
                command[command.index("-task") + 1] = "megablast"
                index = command.index("-query")
                del command[index:index + 2]
                proc = self.run(label, command + ["-query_list", str(self.data / "queries.list"),
                                "-out", str(directory / "batch.out")], sanitize=bool(self.args.sanitizer) and gpu)
                if gpu:
                    for branch in ("small", "megablast"):
                        for phase in ("scan", "extend"):
                            if f"GBLASTN_KERNEL branch={branch} phase={phase} status=completed" not in proc.stderr:
                                raise AssertionError(f"Pipeline {mode} omitted CUDA {branch}/{phase}")
                for query in ("small.fa", "megablast.fa"):
                    path = directory / (query + ".out")
                    if not path.exists() or not rows(path):
                        raise AssertionError(f"Missing pipeline result {path}")
                outputs.append(directory)
            if len(outputs) == 2:
                for query in ("small.fa", "megablast.fa"):
                    compare(outputs[0] / (query + ".out"), outputs[1] / (query + ".out"))
            if mode:
                for query in ("small.fa", "megablast.fa"):
                    compare(self.root / "pipeline-0-cpu" / (query + ".out"), outputs[0] / (query + ".out"))

    def additional_scans(self):
        # These word sizes exercise direct-extension and nondefault small-table
        # dispatch, including the exact-scoring branch for words shorter than 11.
        for word in (7, 9, 12):
            paths = []
            for gpu in ([False, True] if self.args.suite == "gpu" else [False]):
                label = f"word-{word}-{'gpu' if gpu else 'cpu'}"
                out = self.root / (label + ".tsv")
                result = self.run(label, self.command(gpu=gpu) + ["-word_size", str(word), "-out", str(out)],
                                  sanitize=bool(self.args.sanitizer) and gpu)
                paths.append(out)
                if gpu:
                    self.evidence(result.stderr, "small", phases=("scan",) if word == 7 else ("scan", "extend"))
            if len(paths) == 2:
                compare(*paths)

    def multiple_batches(self):
        self.env["BATCH_SIZE"] = "500"
        try:
            for mode in (0, 1, 2):
                for threads in ((1,) if mode == 1 else (1, 2)):
                    for gpu in ([False, True] if self.args.suite == "gpu" else [False]):
                        label = f"multibatch-{mode}-{'gpu' if gpu else 'cpu'}"
                        if threads > 1:
                            label += f"-threads-{threads}"
                        out = self.root / (label + ".tsv")
                        command = self.command(gpu=gpu, mode=mode)
                        command[command.index("-num_threads") + 1] = str(threads)
                        result = self.run(label, command + ["-out", str(out)],
                                          sanitize=bool(self.args.sanitizer) and gpu)
                        submitted = re.findall(r"GBLASTN_BATCH .* index=(\d+) stage=submitted", result.stderr)
                        completed = re.findall(r"GBLASTN_BATCH .* index=(\d+) stage=completed", result.stderr)
                        if len(submitted) < 2 or collections.Counter(submitted) != collections.Counter(completed):
                            raise AssertionError("Multi-batch test did not submit and complete multiple query batches")
                        compare(self.root / "small-both-dust-no-cpu.tsv", out)
        finally:
            self.env.pop("BATCH_SIZE")

    def threaded_searches(self):
        for branch in ("small", "megablast"):
            for gpu in ([False, True] if self.args.suite == "gpu" else [False]):
                label = f"{branch}-threads-2-{'gpu' if gpu else 'cpu'}"
                out = self.root / (label + ".tsv")
                command = self.command(branch, gpu=gpu)
                command[command.index("-num_threads") + 1] = "2"
                self.run(label, command + ["-out", str(out)], sanitize=bool(self.args.sanitizer) and gpu)
                compare(self.root / f"{branch}-both-dust-no-cpu.tsv", out)

    def failures(self):
        for mode in (0, 1, 2):
            for bad in ("nonexistent.list", "empty.list", "missing.list", "duplicate.list", "collision.list", "failed.list", "after-valid-failed.list"):
                command = self.command(mode=mode)
                index = command.index("-query")
                del command[index:index + 2]
                self.run(f"failure-{mode}-{bad}", command + ["-query_list", str(self.data / bad)], fail=True)
        self.run("invalid-mode", self.command(mode=99), fail=True)
        for batch_size in ("0", "-1"):
            self.env["BATCH_SIZE"] = batch_size
            try:
                result = self.run(f"invalid-batch-size-{batch_size}", self.command(), fail=True)
                if "BATCH_SIZE must be a positive integer" not in result.stderr:
                    raise AssertionError("Invalid batch size failed for an unrelated reason")
            finally:
                self.env.pop("BATCH_SIZE")
        original = (self.data / "small.fa").read_bytes()
        self.run("output-overwrites-input", self.command() + ["-out", str(self.data / "small.fa")], fail=True)
        link = self.root / "input-symlink.out"
        if link.is_symlink():
            link.unlink()
        link.symlink_to(self.data / "small.fa")
        self.run("output-symlink-overwrites-input", self.command() + ["-out", str(link)], fail=True)
        list_directory = self.root / "list-symlink"
        list_directory.mkdir(exist_ok=True)
        list_link = list_directory / "small.fa.out"
        if list_link.is_symlink():
            list_link.unlink()
        list_link.symlink_to(self.data / "small.fa")
        command = self.command(mode=2)
        index = command.index("-query")
        del command[index:index + 2]
        self.run("list-output-symlink-overwrites-input", command + [
            "-query_list", str(self.data / "queries.list"),
            "-out", str(list_directory / "batch.out")], fail=True)
        if (self.data / "small.fa").read_bytes() != original:
            raise AssertionError("Failed input/output collision check destroyed the query")
        if self.args.suite == "no-device":
            result = self.run("gpu-without-device", self.command(gpu=True), fail=True)
            if "GPU requested" not in result.stderr:
                raise AssertionError("No-device test failed for a reason other than GPU availability")
        if self.args.suite == "gpu":
            self.env["GBLASTN_GPU_MEMORY_LIMIT_MB"] = "16"
            try:
                result = self.run("gpu-memory-budget", self.command("megablast", True), fail=True)
                if "memory budget" not in result.stderr and "allocation exceeds" not in result.stderr:
                    raise AssertionError("Memory-budget test failed for an unrelated reason")
            finally:
                self.env.pop("GBLASTN_GPU_MEMORY_LIMIT_MB")
        self.run("contradictory-backend", self.command() + ["-gpu_required", "true"], fail=True)

    def execute(self):
        passed = False
        try:
            if self.args.suite == "gpu" and (sys.platform != "linux" or platform.machine() != "aarch64"):
                raise AssertionError("Thor GPU acceptance must run on Linux aarch64")
            self.prepare()
            if self.args.reference:
                version = self.run("reference-version", [self.args.reference, "-version"])
                if "2.2.28+" not in version.stdout + version.stderr:
                    raise AssertionError("--reference must be NCBI BLAST 2.2.28+; newer baseline comparisons are separate")
            self.searches()
            self.additional_scans()
            self.pipelines()
            self.multiple_batches()
            self.threaded_searches()
            self.failures()
            passed = True
        finally:
            (self.root / "results.json").write_text(json.dumps({"passed": passed, "suite": self.args.suite,
                 "system": platform.platform(), "machine": platform.machine(), "runs": self.records,
                 "note": "children_maxrss is the cumulative child high-water mark; not GPU allocation usage"}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("cpu", "no-device", "gpu"), default="cpu")
    parser.add_argument("--work", type=Path, required=True, help="Preserved logs, fixtures, outputs, timing report")
    parser.add_argument("--blastn", default="blastn")
    parser.add_argument("--makeblastdb", default="makeblastdb")
    parser.add_argument("--blastdbcmd", default="blastdbcmd")
    parser.add_argument("--reference", help="Optional independent NCBI 2.2.28+ CPU blastn")
    parser.add_argument("--sanitizer", choices=("memcheck", "racecheck", "synccheck"))
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    if args.sanitizer and args.suite != "gpu":
        parser.error("--sanitizer requires --suite gpu")
    for name in ("blastn", "makeblastdb", "blastdbcmd", "reference"):
        value = getattr(args, name)
        if value:
            resolved = shutil.which(value)
            if not resolved:
                parser.error(f"Executable not found: {value}")
            setattr(args, name, str(Path(resolved).resolve()))
    Acceptance(args).execute()


if __name__ == "__main__":
    main()
