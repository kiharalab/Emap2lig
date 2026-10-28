"""``emap2lig``: find ligand density blobs in a map and model ligands into them."""

from pathlib import Path

import typer

from ..utils.log import setup_logging
from .common import DEFAULTS, make_config, run_stage

app = typer.Typer()


@app.command()
def main(
    input_map: Path = typer.Option(
        ..., "--input-map", help="Path to input cryo-EM map file"
    ),
    ligand_list: Path = typer.Option(
        ..., "--ligand-list", help="YAML list of candidate ligands"
    ),
    output_dir: Path = typer.Option(
        Path("./output"), "--output-dir", help="Directory to save output files"
    ),
    gpu: int = typer.Option(DEFAULTS.gpu, "--gpu", help="Accelerator device ID"),
    emdb_id: str | None = typer.Option(None, "--emdb-id", help="EMDB ID for the map"),
    contour_level: float | None = typer.Option(
        None, "--contour-level", help="Contour level (default: from EMDB or the map)"
    ),
    detection_batch_size: int = typer.Option(
        DEFAULTS.detection_batch_size,
        "--detection-batch-size",
        min=1,
        help="Batch size for detection sliding-window inference",
    ),
    multiplicity: int = typer.Option(
        DEFAULTS.multiplicity,
        "--multiplicity",
        min=1,
        help="Number of conformers per blob per ligand",
    ),
    max_parallel_multiplicity: int = typer.Option(
        DEFAULTS.max_parallel_multiplicity,
        "--max-parallel-multiplicity",
        min=1,
        help="Maximum conformers generated in one forward pass",
    ),
    num_sampling_steps: int = typer.Option(
        DEFAULTS.num_sampling_steps,
        "--num-sampling-steps",
        min=1,
        help="Diffusion sampling steps for structure modeling",
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
) -> None:
    """Detect ligands in cryo-EM density maps using Emap2lig."""
    setup_logging()
    cfg = make_config(
        gpu=gpu,
        contour_level=contour_level,
        detection_batch_size=detection_batch_size,
        multiplicity=multiplicity,
        max_parallel_multiplicity=max_parallel_multiplicity,
        num_sampling_steps=num_sampling_steps,
    )

    from lightning import seed_everything

    from ..data.ligand_list import parse_ligand_list
    from ..pipeline import detect_ligand_objects, run_structure_modeling

    seed_everything(seed)
    ligand_records = parse_ligand_list(ligand_list)
    run_stage(detect_ligand_objects, input_map, output_dir, cfg, emdb_id)
    run_stage(run_structure_modeling, output_dir, ligand_records, cfg)


if __name__ == "__main__":
    app()
