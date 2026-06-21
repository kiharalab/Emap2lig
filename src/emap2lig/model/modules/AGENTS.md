# MODULES KNOWLEDGE BASE

**Generated:** 2026-02-10 (updated 2026-02-12)
**Path:** `src/emap2lig/model/modules/`
**Scope Note:** No functional module changes in the current web UI/documentation update set.

## OVERVIEW

High-level composite modules that assemble primitives from `../layers/` into the Emap2lig pipeline stages: reference embedding, molecular graph processing, EM density conditioning, and EDM-based diffusion coordinate generation.

## STRUCTURE

```
modules/
├── conf_embedder.py     # ConformerEmbedder: input → atom/pair embeddings (90 lines)
├── pairformer.py        # PairFormer + AuxiliaryModule (441 lines)
├── instance_seg.py      # InstanceSegModule: MUNet backbone + InstanceSeg (374 lines)
├── conditioning.py      # AtomConditioner + PointConditioner (376 lines)
├── diffusion.py         # DiffusionModule + AtomDiffusion (510 lines)
└── __init__.py          # Public module exports
```

## KEY COMPONENTS

### ConformerEmbedder (`conf_embedder.py`, 90 lines)
Lightest module. Translates a reference ligand conformer into single-atom and pairwise feature representations. Projects raw atom features (149-dim) and pair features to hidden dims via `LinearNoBias`. Computes relative position vectors and normalized interatomic distances from reference coordinates. Adds atom-to-pair broadcasting.

**I/O**: `(ref_pos[B,N,3], atom_feat[B,N,149], pair_feat[B,N,N,9], masks)` → `(atom[B,N,128], pair[B,N,N,64])`

### PairFormer (`pairformer.py`, lines 21-290)
Central module for chemical feature propagation, following the AlphaFold3 design (reduced to four blocks). Stack of `PairFormerBlock` layers. Each block runs 7 operations sequentially with residual connections:
1. `AttentionPairBias` — pair-biased multi-head attention on atoms
2. `OuterProductMean` — atom→pair interaction projection
3. `TriangleMultiplicationOutgoing/Incoming` — pair triangular consistency
4. `TriangleAttentionStarting/EndingNode` — axial pair attention
5. `Transition` — gated MLP on atoms and pairs separately

Optional `cuequivariance` CUDA kernels and `fairscale` activation checkpointing.

### AuxiliaryModule (`pairformer.py`, lines 292-441)
Auxiliary prediction heads from PairFormer output:
- **Distogram**: pair_feats → `[B,N,N,num_bins]` distance bin logits
- **Element**: atom_feats → 6-class element prediction
- **Chirality**: atom_feats → 7-class chirality prediction
- **Bond type**: pair_feats → 5-class bond type prediction
- **Ring size**: pair_feats → 4-class ring membership prediction

### InstanceSegModule (`instance_seg.py`, 374 lines)
Instance segmentation module that predicts a ligand-specific 3D probability mask and bridges cryo-EM density to atom representations. Composition:
- `MUNetBackbone` — 3D U-Net extracting multi-scale voxel features
- `SegHead` — 15-channel segmentation probability map (sigmoid)
- `InstanceSeg` — cross-attention between voxel grid and atom queries
- `VoxelProjector` — global average pooling → aggregated voxel features

**Two-phase forward**:
1. `forward_embedding()`: density → backbone features → seg map → instance selection of top-8192 points
2. Integration into diffusion conditioning via `PointConditioner`

### AtomConditioner (`conditioning.py`, lines 22-115)
Per-step diffusion conditioning on atom features:
- Concatenates current + initial atom representations
- Adds `PositionalEncoder` from noisy coordinates
- Adds `FourierEmbedding` of diffusion timestep σ
- Refines with `Transition` layers

### PointConditioner (`conditioning.py`, lines 118-376)
Stack of `PointConditionerBlock` modules performing:
1. `SelectedCrossAttention` — atoms attend to pre-selected EM density points with 3D positional encoding
2. `AttentionPairBias` — self-attention among atoms with pair bias
3. `Transition` — gated MLP refinement

### AtomDiffusion (`diffusion.py`, lines 300-510)
EDM (Elucidating Diffusion Models) sampling wrapper:
- **Schedule**: Karras noise schedule with `rho=7.0`, `sigma_min=0.0004`, `sigma_max=160.0`
- **Preconditioning**: `c_skip`, `c_out`, `c_in` functions of σ with `sigma_data=16.0`
- **Sampling**: 20-step Euler-Heun with stochastic churn (`gamma_0=0.8`)
- **Network**: `DiffusionModule` (lines 52-297) wraps AtomConditioner + PointConditioner + AtomDecoder

### DiffusionModule (`diffusion.py`, lines 52-297)
The denoising network called at each diffusion step:
1. `AtomConditioner` — time/feature conditioning
2. `PointConditioner` — EM density cross-attention (N blocks)
3. `AtomDecoder` — LayerNorm + Linear → Δcoordinates

## DEPENDENCIES

```
layers/ provides:
  AttentionPairBias, TriangleAttention*, TriangleMult*, OuterProductMean,
  Transition, SelectedCrossAttention, InstanceSeg, AtomDecoder,
  FourierEmbedding, PositionalEncoder, LinearNoBias, LayerNorm, MLP

seg/ provides:
  MUNetBackbone, SegHead

External:
  fairscale (activation checkpointing), einops (rearrange)
```

## ANTI-PATTERNS

- `fairscale` import may fail on systems without it — `pairformer.py` and `conditioning.py` import at top level
