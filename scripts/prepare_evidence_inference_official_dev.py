#!/usr/bin/env python3
"""Extract the full official Evidence Inference 2.0 validation split from its release archive."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import tarfile
from pathlib import Path


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_csv(data: bytes) -> tuple[list[str], list[dict[str, str]]]:
    stream = io.StringIO(data.decode("utf-8-sig"), newline="")
    reader = csv.DictReader(stream)
    return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(buffer.getvalue(), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path,
                        default=Path("data/evidence_inference_full_release/evidence-inference-v2.0-ranged.tar.gz"))
    parser.add_argument("--out", type=Path, default=Path("data/evidence_inference_official_dev"))
    args = parser.parse_args()
    archive_bytes_hash = digest(args.archive.read_bytes())
    expected_hash = "6abe0d4ec0d331834981c0171c3c79d47515761867f82f1dc6066e43863a1586"
    if archive_bytes_hash != expected_hash:
        raise ValueError(f"Release checksum mismatch: {archive_bytes_hash}")

    with tarfile.open(args.archive, "r:gz") as archive:
        def member_bytes(name: str) -> bytes:
            member = archive.extractfile(name)
            if member is None:
                raise FileNotFoundError(name)
            return member.read()

        split_data = {
            split: member_bytes(f"splits/{split}_article_ids.txt")
            for split in ("train", "validation", "test")
        }
        split_ids = {
            split: {line.strip() for line in data.decode("utf-8").splitlines() if line.strip()}
            for split, data in split_data.items()
        }
        if (split_ids["train"] & split_ids["validation"] or
                split_ids["train"] & split_ids["test"] or
                split_ids["validation"] & split_ids["test"]):
            raise ValueError("The release's generic article splits are not mutually disjoint")

        prompt_fields, all_prompts = read_csv(member_bytes("prompts_merged.csv"))
        annotation_fields, all_annotations = read_csv(member_bytes("annotations_merged.csv"))
        validation_prompts = [row for row in all_prompts
                              if row.get("PMCID", "").strip() in split_ids["validation"]]
        article_ids = sorted({row.get("PMCID", "").strip() for row in validation_prompts})
        if not article_ids:
            raise ValueError("No prompts joined to the official validation split")
        validation_set = set(article_ids)
        if validation_set & split_ids["train"] or validation_set & split_ids["test"]:
            raise ValueError("Joined validation articles overlap train or test")
        annotations = [row for row in all_annotations
                       if row.get("PMCID", "").strip() in validation_set]

        args.out.mkdir(parents=True, exist_ok=True)
        articles_dir = args.out / "articles"
        articles_dir.mkdir(parents=True, exist_ok=True)
        for split, ids in split_ids.items():
            (args.out / f"{split}_article_ids.txt").write_bytes(split_data[split])
        write_csv(args.out / "prompts_validation.csv", prompt_fields, validation_prompts)
        write_csv(args.out / "annotations_validation.csv", annotation_fields, annotations)

        article_hashes: dict[str, str] = {}
        accepted_rows = 0
        valid_offset_rows = 0
        in_bounds_rows = 0
        accepted_in_bounds_rows = 0
        article_texts: dict[str, str] = {}
        for pmcid in article_ids:
            raw = member_bytes(f"txt_files/PMC{pmcid}.txt")
            (articles_dir / f"PMC{pmcid}.txt").write_bytes(raw)
            article_hashes[pmcid] = digest(raw)
            article_texts[pmcid] = raw.decode("utf-8")

        prompt_ids = {row.get("PromptID", "").strip() for row in validation_prompts}
        valid_prompt_ids: set[str] = set()
        annotator_groups: set[tuple[str, str]] = set()
        for row in annotations:
            accepted = (row.get("Valid Label", "").lower() == "true" and
                        row.get("Valid Reasoning", "").lower() == "true")
            if accepted:
                accepted_rows += 1
            try:
                start = int(row.get("Evidence Start", ""))
                end = int(row.get("Evidence End", ""))
            except (TypeError, ValueError):
                continue
            if start < 0 or end < start:
                continue
            valid_offset_rows += 1
            pmcid = row.get("PMCID", "").strip()
            if end >= len(article_texts[pmcid]):
                continue
            in_bounds_rows += 1
            if accepted:
                accepted_in_bounds_rows += 1
                valid_prompt_ids.add(row.get("PromptID", "").strip())
                annotator_groups.add((row.get("PromptID", "").strip(), row.get("UserID", "").strip()))

        (args.out / "release_README.txt").write_bytes(member_bytes("README.md"))
        manifest = {
            "dataset": "Evidence Inference 2.0 official full validation/development split",
            "release_url": "https://evidence-inference.ebm-nlp.com/v2.0.tar.gz",
            "release_sha256": archive_bytes_hash,
            "split_semantics": "generic train/validation/test lists in the official release are the unions of the EV1 and EV2 cohort-specific lists; EV2-only files are subsets, not the full official split",
            "official_split_id_counts": {name: len(ids) for name, ids in split_ids.items()},
            "joined_validation_articles_with_prompts": len(article_ids),
            "joined_validation_prompt_rows": len(validation_prompts),
            "unique_validation_prompt_ids": len(prompt_ids),
            "joined_validation_annotation_rows": len(annotations),
            "accepted_valid_label_and_reasoning_rows": accepted_rows,
            "valid_nonnegative_offset_rows": valid_offset_rows,
            "in_bounds_offset_rows": in_bounds_rows,
            "accepted_in_bounds_rows": accepted_in_bounds_rows,
            "unique_prompts_with_accepted_in_bounds_rationale": len(valid_prompt_ids),
            "accepted_prompt_annotator_groups": len(annotator_groups),
            "validation_overlap_train": len(validation_set & split_ids["train"]),
            "validation_overlap_test": len(validation_set & split_ids["test"]),
            "previously_inspected_ev2_test_overlap": len(
                validation_set & {line.strip() for line in
                                  (Path("data/evidence_inference/ev2_test_article_ids.txt").read_text().splitlines())
                                  if line.strip()}),
            "hashes": {
                "split_files": {name: digest(data) for name, data in split_data.items()},
                "prompts_csv": digest(member_bytes("prompts_merged.csv")),
                "annotations_csv": digest(member_bytes("annotations_merged.csv")),
                "validation_prompts_csv": digest((args.out / "prompts_validation.csv").read_bytes()),
                "validation_annotations_csv": digest((args.out / "annotations_validation.csv").read_bytes()),
                "validation_articles": article_hashes,
            },
            "redistribution": "Annotation/code license is separate from article text; do not redistribute full text without checking each article's PMC license.",
        }
        (args.out / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({key: value for key, value in manifest.items() if key != "hashes"},
                         indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
