#!/usr/bin/env python3
"""Prepare unused official EI 2.0 test articles, excluding the previously inspected EV2 subset."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import tarfile
from pathlib import Path


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_csv(data: bytes) -> tuple[list[str], list[dict[str, str]]]:
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline=""))
    return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(buffer.getvalue(), encoding="utf-8")


def ids_from_bytes(data: bytes) -> set[str]:
    return {line.strip() for line in data.decode("utf-8").splitlines() if line.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path,
                        default=Path("data/evidence_inference_full_release/evidence-inference-v2.0-ranged.tar.gz"))
    parser.add_argument("--previously-inspected-ids", type=Path,
                        default=Path("data/evidence_inference/ev2_test_article_ids.txt"))
    parser.add_argument("--out", type=Path, default=Path("data/evidence_inference_untouched_test"))
    args = parser.parse_args()
    archive_hash = sha256(args.archive.read_bytes())
    expected = "6abe0d4ec0d331834981c0171c3c79d47515761867f82f1dc6066e43863a1586"
    if archive_hash != expected:
        raise ValueError(f"Release checksum mismatch: {archive_hash}")
    previous_ids = ids_from_bytes(args.previously_inspected_ids.read_bytes())

    with tarfile.open(args.archive, "r:gz") as archive:
        def member_bytes(name: str) -> bytes:
            stream = archive.extractfile(name)
            if stream is None:
                raise FileNotFoundError(name)
            return stream.read()

        split_data = {
            name: member_bytes(f"splits/{name}_article_ids.txt")
            for name in ("train", "validation", "test")
        }
        split_ids = {name: ids_from_bytes(data) for name, data in split_data.items()}
        if any(split_ids[a] & split_ids[b] for a, b in (
                ("train", "validation"), ("train", "test"), ("validation", "test"))):
            raise ValueError("Official generic article splits are not disjoint")
        if previous_ids - split_ids["test"]:
            raise ValueError("Previously inspected EV2 IDs are not a subset of the official test split")

        prompt_fields, all_prompts = read_csv(member_bytes("prompts_merged.csv"))
        annotation_fields, all_annotations = read_csv(member_bytes("annotations_merged.csv"))
        selected_test_ids = split_ids["test"] - previous_ids
        test_prompts = [row for row in all_prompts
                        if row.get("PMCID", "").strip() in selected_test_ids]
        article_ids = sorted({row.get("PMCID", "").strip() for row in test_prompts})
        article_set = set(article_ids)
        if article_set & split_ids["train"] or article_set & split_ids["validation"]:
            raise ValueError("Selected test articles overlap train or validation")
        if article_set & previous_ids:
            raise ValueError("Selected test articles overlap previously inspected subset")
        test_annotations = [row for row in all_annotations
                            if row.get("PMCID", "").strip() in article_set]

        args.out.mkdir(parents=True, exist_ok=True)
        articles_dir = args.out / "articles"
        articles_dir.mkdir(parents=True, exist_ok=True)
        for name, data in split_data.items():
            (args.out / f"{name}_article_ids.txt").write_bytes(data)
        (args.out / "excluded_previously_inspected_ev2_test_article_ids.txt").write_bytes(
            args.previously_inspected_ids.read_bytes())
        write_csv(args.out / "prompts.csv", prompt_fields, test_prompts)
        write_csv(args.out / "annotations.csv", annotation_fields, test_annotations)

        article_hashes = {}
        for pmcid in article_ids:
            data = member_bytes(f"txt_files/PMC{pmcid}.txt")
            (articles_dir / f"PMC{pmcid}.txt").write_bytes(data)
            article_hashes[pmcid] = sha256(data)

        manifest = {
            "dataset": "Evidence Inference 2.0 official test split, previously unseen article subset",
            "release_sha256": archive_hash,
            "split_list_counts": {name: len(ids) for name, ids in split_ids.items()},
            "full_test_articles_with_prompts": len({
                row.get("PMCID", "").strip() for row in all_prompts
                if row.get("PMCID", "").strip() in split_ids["test"]
            }),
            "previously_inspected_ev2_test_articles_excluded": len(previous_ids),
            "remaining_test_article_ids": len(selected_test_ids),
            "remaining_test_articles_with_prompts": len(article_ids),
            "remaining_prompt_rows": len(test_prompts),
            "remaining_unique_prompt_ids": len({row.get("PromptID", "").strip() for row in test_prompts}),
            "remaining_annotation_rows": len(test_annotations),
            "overlap_train": len(article_set & split_ids["train"]),
            "overlap_validation": len(article_set & split_ids["validation"]),
            "overlap_previously_inspected": len(article_set & previous_ids),
            "hashes": {
                "release_train_split": sha256(split_data["train"]),
                "release_validation_split": sha256(split_data["validation"]),
                "release_test_split": sha256(split_data["test"]),
                "excluded_ev2_test_ids": sha256(args.previously_inspected_ids.read_bytes()),
                "prompts_csv": sha256(member_bytes("prompts_merged.csv")),
                "annotations_csv": sha256(member_bytes("annotations_merged.csv")),
                "prepared_prompts_csv": sha256((args.out / "prompts.csv").read_bytes()),
                "prepared_annotations_csv": sha256((args.out / "annotations.csv").read_bytes()),
                "articles": article_hashes,
            },
            "redistribution": "Do not redistribute full text without checking each article's PMC reuse license.",
        }
        (args.out / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({key: value for key, value in manifest.items() if key != "hashes"},
                         indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
