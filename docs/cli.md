# CLI Reference

## Main command

```bash
emap2lig --input-map <MAP> --output-dir <OUTPUT_DIR> --ligand-list <LIGANDS_YAML> [OPTIONS]
```

From a cloned repo, prefix with `uv run`:

```bash
uv run emap2lig --input-map <MAP> --output-dir <OUTPUT_DIR> --ligand-list <LIGANDS_YAML> [OPTIONS]
```

Install options: [Installation](installation.md).

## Required arguments

| Argument | Description |
|----------|-------------|
| `--input-map` | Path to cryo-EM map (`.map.gz`, `.mrc`) — see [Input formats](input-format.md#cryo-em-map) |
| `--output-dir` | Output directory (default: `./output`) |
| `--ligand-list` | Ligand YAML — see [Input formats](input-format.md#ligand-list-yaml) |

## Optional arguments

| Argument | Description |
|----------|-------------|
| `--gpu` | Accelerator device ID (default: `0`). On Linux this is the CUDA device ID; on macOS MPS exposes a single device (`0`). |
| `--detection-batch-size` | Sliding-window batch size for detection (default config: `16`) |
| `--emdb-id` | EMDB ID for automatic contour level lookup |
| `--contour-level` | Manual contour level (overrides EMDB lookup) |
| `--multiplicity` | Conformers per ligand–blob pair (default: `1`) |
| `--max-parallel-multiplicity` | Maximum conformers generated in one forward pass (default: `8`). Lower this to reduce peak GPU memory when using large `--multiplicity`. |
| `--seed` | Random seed (default: `42`) |

Local CLI inference supports Linux/CUDA and macOS/MPS. CPU inference is not
supported.

Use `--emdb-id` or `--contour-level` for map normalization — see
[Map normalization](input-format.md#map-normalization).

## Examples

```bash
# Simple ligands (EMD-30556)
uv run emap2lig \
  --input-map examples/emd_30556.map.gz \
  --gpu 0 \
  --emdb-id 30556 \
  --detection-batch-size 4 \
  --output-dir outputs_30556 \
  --ligand-list examples/emd_30556.yaml

# Branched ligands (EMD-7783)
uv run emap2lig \
  --input-map examples/emd_7783.map.gz \
  --gpu 0 \
  --emdb-id 7783 \
  --detection-batch-size 4 \
  --output-dir outputs_7783 \
  --ligand-list examples/emd_7783.yaml \
  --multiplicity 1
```

For large conformer searches, keep `--multiplicity` as the total number of
conformers to generate and lower `--max-parallel-multiplicity` if GPU memory is
limited. Outputs are still written together and ranked by `consistency_iou`.

## Fragment detection

Lightweight fragment-class maps only (no full Build stage):

```bash
uv run fragment-detect --input-map <MAP> [OPTIONS]
```

Details: [Fragment detection](fragment-detection.md).

## Output layout

See [Output structure](output.md).
