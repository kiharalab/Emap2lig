# Fragment Detection

Optional **Find-only** mode for fragment-class probability maps (e.g. 5- and
6-membered rings) without full ligand building.

## Command

```bash
fragment-detect --input-map <MAP> [OPTIONS]
```

From a clone: `uv run fragment-detect ...`

## Options

| Option | Description |
|--------|-------------|
| `--output-dir` | Output directory (default: `./output`) |
| `--gpu` | CUDA device ID (default: `0`) |
| `--detection-batch-size` | Sliding-window batch size |
| `--emdb-id` | EMDB ID for contour level |
| `--contour-level` | Manual contour level |

Map inputs: [Input formats](input-format.md).

## Outputs

```text
{stem}_frag_{label}.mrc        # probability map per fragment class
{stem}_frag_{label}_mask.mrc   # binary mask per fragment class
```

Model: `emap2lig-frag.safetensors` on [HuggingFace](models.md).
