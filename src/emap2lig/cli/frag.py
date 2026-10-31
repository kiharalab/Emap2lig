"""``fragment-detect``: per-voxel ring/fragment segmentation of a map."""

from pathlib import Path

import typer

from ..utils.log import setup_logging
from .common import DEFAULTS, make_config, make_tqdm_callback, run_stage

app = typer.Typer()


@app.command()
def main(
    input_map: Path = typer.Option(
        ..., "--input-map", help="Path to input cryo-EM map file"
    ),
    output_dir: Path = typer.Option(
        Path("./output"), "--output-dir", help="Directory to save outputs"
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
        help="Batch size for sliding-window inference",
    ),
) -> None:
    """Run fragment detection using FragmentRegSeg and save predictions."""
    setup_logging()
    cfg = make_config(
        gpu=gpu, contour_level=contour_level, detection_batch_size=detection_batch_size
    )

    from ..pipeline import run_fragment_detection

    run_stage(
        run_fragment_detection,
        input_map,
        output_dir,
        cfg,
        emdb_id,
        on_progress=make_tqdm_callback(),
    )


if __name__ == "__main__":
    app()
