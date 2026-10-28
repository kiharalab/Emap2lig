"""Rank the conformers of each blob by consistency IoU and collect the best."""

import csv
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def rank_conformers(build_struct_dir: Path, best_dir: Path) -> None:
    """Sort each ``blob_*/blob_*_results.csv`` by score; copy each top CIF to *best_dir*.

    The writer appends one row per conformer during inference; CIF files that
    lack a row are ranked with score 0.
    """
    best_dir.mkdir(parents=True, exist_ok=True)
    for blob_dir in build_struct_dir.iterdir():
        if blob_dir.is_dir() and blob_dir.name.startswith("blob_"):
            _rank_blob(blob_dir, best_dir)


def _rank_blob(blob_dir: Path, best_dir: Path) -> None:
    csv_file = blob_dir / f"{blob_dir.name}_results.csv"
    scores = _read_scores(csv_file)
    for cif_file in blob_dir.glob("*.cif"):
        scores.setdefault(cif_file.stem, 0.0)
    if not scores:
        logger.warning(f"No conformers found in {blob_dir}")
        return

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["conformer_name", "consistency_iou"])
        writer.writerows(ranked)
    logger.info(f"Ranked {len(ranked)} conformers in {csv_file}")

    best_name, best_score = ranked[0]
    best_cif = blob_dir / f"{best_name}.cif"
    if not best_cif.exists():
        logger.warning(f"Best .cif file not found: {best_cif}")
        return
    dest = best_dir / f"{blob_dir.name}_{best_name}.cif"
    try:
        shutil.copy2(best_cif, dest)
        logger.info(f"Copied best result: {dest} (score: {best_score:.4f})")
    except OSError as e:
        logger.warning(f"Failed to copy {best_cif} to best directory: {e}")


def _read_scores(csv_file: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    if not csv_file.exists():
        return scores
    try:
        with open(csv_file) as f:
            for row in csv.DictReader(f):
                try:
                    scores[row["conformer_name"]] = float(row["consistency_iou"])
                except (KeyError, ValueError):
                    pass
    except OSError as e:
        logger.warning(f"Failed to read CSV {csv_file}: {e}")
    return scores
