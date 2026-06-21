# SEGMENTATION KNOWLEDGE BASE

**Generated:** 2026-01-12 (updated 2026-02-12)
**Path:** `src/emap2lig/model/seg/`
**Scope Note:** No functional segmentation changes in the current web UI update set.

## OVERVIEW
Multi-class 3D segmentation and regression framework for EM maps using transformer-augmented U-Nets.

## STRUCTURE
- `model.py`: Main model definitions and inference orchestration
  - `MUNetRegSeg`: Primary dual-head architecture
  - `FragmentRegSeg`: Staged backbone for specialized fragment detection
- `threshold.py`: Iterative Li thresholding for class-specific binarization
- `munet/`: Transformer-based 3D U-Net backbone implementation
  - `backbone.py`: Hierarchical encoder-decoder with skip connections
  - `transformer.py`: Multi-head 3D spatial self-attention blocks
  - `conv.py`: 3D ResBlocks with SiLU and GroupNorm, plus pooling primitives

## KEY COMPONENTS
- **3D Transformer U-Net**: Hybrid architecture combining local CNN features with long-range transformer-based spatial reasoning.
- **Multi-task Heads**:
  - `SegHead`: Multi-label segmentation (GroupNorm + SiLU + Conv3d).
  - `RegHead`: Multi-channel regression for spatial offsets/features.
- **Target Classes**:
  - **Structural**: `backbone`, `sidechain`, `sugar`, `phosphate`, `base`, `ligand`.
  - **Fragments**: `5R` (5-membered ring), `6R`, `4H` (4-heavy atoms), `5H`, `DR`, `HEM-like`.
- **Large-Volume Inference**:
  - `SlidingWindowInference`: Chunks large maps into overlapping ROIs (default 64^3).
  - `Uniform ROI Averaging`: Merges overlapping ROI predictions by simple count-based averaging.
  - `VolumeInferenceDataset`: Optimized sub-volume sampling and coordinate tracking.
- **Post-Processing**:
  - `Li Thresholding`: Per-channel adaptive thresholding for robust mask generation.
  - `tensor_to_point_cloud`: Voxel-to-point conversion with global coordinate scaling.
  - `get_centered_point_cloud`: ROI cropping and feature fusion for AtomDiffusion input.
