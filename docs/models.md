# Model Weights (HuggingFace Hub)

Repository: [huggingface.co/KiharaLab/Emap2lig](https://huggingface.co/KiharaLab/Emap2lig)

Files **download automatically** on first run to `~/.emap2lig/models/` via `huggingface_hub` (network required).

| Resource | HF file | Used by |
|----------|---------|---------|
| Detection model | `emap2lig-find-v0.0.1.safetensors` | `MUNetRegSeg` (map segmentation) |
| Fragment model | `emap2lig-frag.safetensors` | `FragmentRegSeg` |
| Structure model | `emap2lig-build-v0.0.1.safetensors` | `Emap2lig` (diffusion modeling) |
| CCD dictionary | `ccd/ccd_dict_250523.pkl` | Reference conformers |
| License | `LICENSE.md` | Academic and Non-Profit Research License (see [WEIGHT_LICENSE.md](../WEIGHT_LICENSE.md)) |

## License

- **Source code** (this repo): [GPL-3.0](../LICENSE)
- **Model weights**: [WEIGHT_LICENSE.md](../WEIGHT_LICENSE.md) (canonical text on [Hugging Face](https://huggingface.co/KiharaLab/Emap2lig/blob/main/LICENSE.md))

Commercial use of the model weights is not permitted without permission.
For commercial licensing inquiries, please contact the authors.

## Syncing updates (maintainers)

When releasing new weights or CCD data:

1. Upload to [KiharaLab/Emap2lig](https://huggingface.co/KiharaLab/Emap2lig).
2. Update `filename` / `repo_id` in `src/emap2lig/emap2lig.yaml` (`detection_model`, `fragment_detection_model`, `model`).
3. Update `REPO_ID` in `src/emap2lig/main.py` and `repo_id` in `src/emap2lig/data/ccd.py`.
4. Update the CCD date in `get_ccd_dict(date="...")` in `src/emap2lig/data/ccd.py` when applicable.
