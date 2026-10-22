#!/usr/bin/env python3
"""Fetch original public datasets and the upstream BGE checkpoint.

Downloaded datasets and model weights are intentionally not part of the Git
repository. Check provider terms before use; Evidence Inference article text
may have article-specific reuse restrictions.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import tarfile
import urllib.request
from pathlib import Path


ASSETS = {
    "qasper_test": {
        "url": "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz",
        "path": Path("data/archives/qasper-test-and-evaluator-v0.3.tgz"),
        "extract_to": Path("data/qasper_test"),
        "sha256": None,
    },
    "evidence_inference": {
        "url": "https://evidence-inference.ebm-nlp.com/v2.0.tar.gz",
        "path": Path("data/archives/evidence-inference-v2.0-ranged.tar.gz"),
        "extract_to": None,
        "sha256": "6abe0d4ec0d331834981c0171c3c79d47515761867f82f1dc6066e43863a1586",
    },
}
MODEL_ID = "BAAI/bge-reranker-v2-m3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_extract(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive_path, "r:*") as archive:
        members = archive.getmembers()
        for member in members:
            if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                raise RuntimeError(f"refusing non-regular archive member: {member.name}")
            target = (destination / member.name).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"refusing archive member outside destination: {member.name}")
        archive.extractall(destination, members=members)


def fetch_dataset(name: str) -> None:
    spec = ASSETS[name]
    path = spec["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"Downloading {name} from {spec['url']}")
        with urllib.request.urlopen(spec["url"]) as response, path.open("wb") as output:
            shutil.copyfileobj(response, output)
    digest = sha256(path)
    expected = spec["sha256"]
    if expected and digest != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, got {digest}")
    print(f"{path}: sha256={digest}")
    destination = spec["extract_to"]
    if destination is not None and not destination.exists():
        safe_extract(path, destination)
        print(f"Extracted archive under {destination}")


def fetch_bge() -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("Install requirements first: python -m pip install -r requirements.txt") from exc
    destination = Path("models/bge-reranker-v2-m3")
    print(f"Downloading {MODEL_ID} to {destination}")
    snapshot_download(repo_id=MODEL_ID, local_dir=str(destination))
    weights = destination / "model.safetensors"
    if weights.exists():
        print(f"{weights}: sha256={sha256(weights)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include", nargs="+", required=True,
                        choices=[*ASSETS, "bge", "all"],
                        help="Assets to download; 'all' downloads both datasets and BGE weights.")
    args = parser.parse_args()
    selected = [*ASSETS, "bge"] if "all" in args.include else args.include
    for name in dict.fromkeys(selected):
        if name == "bge":
            fetch_bge()
        else:
            fetch_dataset(name)


if __name__ == "__main__":
    main()
