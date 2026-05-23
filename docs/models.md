# Model Weights (HuggingFace Hub)

Repository: [huggingface.co/KiharaLab/Emap2lig](https://huggingface.co/KiharaLab/Emap2lig)

Files **download automatically** on first run to `~/.emap2lig/models/` via `huggingface_hub` (network required).

| Resource | HF file | Used by |
|----------|---------|---------|
| Detection model | `emap2lig-find-v0.0.1.safetensors` | `MUNetRegSeg` (map segmentation) |
| Fragment model | `emap2lig-frag.safetensors` | `FragmentRegSeg` |
| Structure model | `emap2lig-build-v0.0.1.safetensors` | `Emap2lig` (diffusion modeling) |
| CCD dictionary | `ccd/ccd_dict_250523.pkl` | Reference conformers |
| License | `LICENSE.md` | Academic and Non-Profit Research License |

## License

> [!IMPORTANT]
> **Model weights** use a custom **Academic and Non-Profit Research License** —
> separate from the code license.
>
> - **Source code** (this repo): [GPL-3.0](../LICENSE)
> - **Model weights** (HuggingFace): [Academic and Non-Profit Research License](https://huggingface.co/KiharaLab/Emap2lig/blob/main/LICENSE.md)
>
> Weights are **not** GPL-3.0. Review both licenses before inference or
> fine-tuning.

## Syncing updates (maintainers)

When releasing new weights or CCD data:

1. Upload to [KiharaLab/Emap2lig](https://huggingface.co/KiharaLab/Emap2lig).
2. Update `filename` / `repo_id` in `src/emap2lig/emap2lig.yaml` (`detection_model`, `fragment_detection_model`, `model`).
3. Update `REPO_ID` in `src/emap2lig/main.py` and `repo_id` in `src/emap2lig/data/ccd.py`.
4. Update the CCD date in `get_ccd_dict(date="...")` in `src/emap2lig/data/ccd.py` when applicable.
