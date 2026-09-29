#!/usr/bin/env python3
"""Deterministic, redistributable fixtures; no downloads or third-party modules."""
import argparse
import hashlib
import json
import random
from pathlib import Path


def fasta(path, records):
    with path.open("w") as stream:
        for name, sequence in records:
            stream.write(f">{name}\n")
            for offset in range(0, len(sequence), 80):
                stream.write(sequence[offset:offset + 80] + "\n")


def generate(directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    rng = random.Random(228110)
    dna = lambda length: "".join(rng.choice("ACGT") for _ in range(length))
    reverse = lambda sequence: sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    exact, substitute, gap, ambiguous, rev, tail, nohit = [dna(240) for _ in range(7)]
    masked = dna(80) + "ac" * 40 + dna(80)
    queries = [("exact", exact), ("substitutions", substitute), ("gap", gap),
               ("ambiguous", ambiguous), ("reverse", rev), ("tail", tail),
               ("masked", masked), ("nohit", nohit)]
    subjects = [exact, substitute[:100] + "AACCG" + substitute[105:],
                gap[:110] + "GATTACA" + gap[110:],
                ambiguous[:100] + "NNNNN" + ambiguous[105:], reverse(rev), masked.upper()]
    # More than 8,500 indexed words on each strand selects the MB lookup
    # in this exact baseline. Task=megablast with tiny queries selects small-NA!
    pad = dna(10000)
    fasta(directory / "small.fa", queries)
    fasta(directory / "megablast.fa", queries + [("mb_lookup_padding", pad)])
    datasets = {}
    for branch, threshold in (("small", 500000), ("megablast", 10000)):
        records = []
        for length in (threshold - 1, threshold, threshold + 1, threshold + 3):
            sequence = list(dna(length))
            for i, insert in enumerate(subjects):
                start = 400 + i * 800
                sequence[start:start + len(insert)] = insert
            # Exercise packed-byte tails at all length mod 4 values.
            sequence[-len(tail):] = tail
            records.append((f"{branch}_{length}", "".join(sequence)))
        fasta(directory / f"{branch}_subjects.fa", records)
        datasets[branch] = {"threshold": threshold, "subject_lengths": [len(s) for _, s in records],
                            "query": f"{branch}.fa", "subjects": f"{branch}_subjects.fa"}
    # Separate files prove the threshold applies per subject, not total database length.
    for branch in datasets:
        source = directory / f"{branch}_subjects.fa"
        pieces = source.read_text().split(">")[1:]
        for label, piece in zip(("below", "at", "above", "tail"), pieces):
            (directory / f"{branch}_{label}.fa").write_text(">" + piece)
    (directory / "queries.list").write_text(f"{directory / 'small.fa'}\n{directory / 'megablast.fa'}\n")
    (directory / "empty.list").write_text("")
    (directory / "missing.list").write_text(f"{directory / 'does-not-exist.fa'}\n")
    (directory / "invalid.fa").write_text(">invalid\n!@#$%^&*()\n")
    (directory / "failed.list").write_text(f"{directory / 'invalid.fa'}\n{directory / 'small.fa'}\n")
    (directory / "after-valid-failed.list").write_text(f"{directory / 'small.fa'}\n{directory / 'invalid.fa'}\n")
    (directory / "duplicate.list").write_text(f"{directory / 'small.fa'}\n{directory / 'small.fa'}\n")
    for folder in ("collision-a", "collision-b"):
        (directory / folder).mkdir(exist_ok=True)
        fasta(directory / folder / "same.fa", [("exact", exact)])
    (directory / "collision.list").write_text(
        f"{directory / 'collision-a/same.fa'}\n{directory / 'collision-b/same.fa'}\n")
    manifest = {"seed": 228110, "datasets": datasets,
                "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(directory.glob("*.fa"))}}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    generate(parser.parse_args().directory)
