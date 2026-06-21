# MODEL KNOWLEDGE BASE

**Generated:** 2026-02-10 (updated 2026-05-22)
**Path:** `src/emap2lig/model/`
**Scope Note:** Updated to reflect current codebase state.

## OVERVIEW

Two-stage neural architecture: ConformerEmbedder encodes molecular graphs → PairFormer processes pairwise interactions → InstanceSegModule extracts voxel features from density → AtomDiffusion generates 3D coordinates via EDM diffusion. All config args defined as dataclasses in `model.py`.

## STRUCTURE

```
model/
├── model.py             # Emap2lig LightningModule + 6 arg dataclasses (461 lines)
├── __init__.py          # Exports: Emap2lig + all *Args dataclasses
├── modules/             # High-level composite modules (see modules/AGENTS.md)
│   ├── diffusion.py     # AtomDiffusion: EDM sampling loop (509 lines)
│   ├── pairformer.py    # PairFormer + AuxiliaryModule (440 lines)
│   ├── instance_seg.py  # InstanceSegModule: density → voxel features (355 lines)
│   ├── conf_embedder.py # ConformerEmbedder: ligand input embedding (90 lines)
│   ├── conditioning.py  # AtomConditioner + PointConditioner (372 lines)
│   └── __init__.py      # Public module exports
├── layers/              # Primitive operations (see layers/AGENTS.md)
└── seg/                 # MUNet segmentation (see seg/AGENTS.md)
```

## COMPOSITION

```
Emap2lig (LightningModule)
├── conf_embedder: ConformerEmbedder      # atom/pair feature → d_atom/d_pair
├── pairformer: PairFormer                # N × PairFormerBlock (triangular attention)
├── auxiliary_module: AuxiliaryModule      # distogram, element, chirality, bond, ring heads
├── instance_seg: InstanceSegModule        # MUNetBackbone + InstanceSeg → voxel features
└── diffusion_module: AtomDiffusion        # EDM loop → sampled coordinates
```

## FORWARD PASS

```
feats = {ref_pos, atom_feature, bond_feature, input_density, ...}
    ↓
ConformerEmbedder → atom_init [B,N,128], pair_init [B,N,N,64]
    ↓
PairFormer (N blocks) → atom_feats, pair_feats
    ↓
AuxiliaryModule → distogram, element, chirality, atom_ring_size,
                  bond_type, bond_ring_size, bond_exists (7 heads)
    ↓
InstanceSegModule → voxel_features [B*M,64,48³],
                     selected_point_feats [B*M,8192,64],
                     selected_point_coords [B*M,8192,3],
                     global_features [B*M,1,256]
    ↓
AtomDiffusion.sample() (20 steps) → sampled_coords [B×M, N_atoms, 3]
```

Notes:
- `B`: batch size, `N`: max atoms, `M`: multiplicity
- Inputs are passed as a single dict (not individual arguments)
- Default dimensions from config (d_atom_hidden=128, d_pair_hidden=64)

## CONFIG ARGS (model.py dataclasses)

| Dataclass | Key Fields | Used By |
|-----------|-----------|---------|
| `ConformerEmbedderArgs` | atom_dim_in, pair_dim_in, atom_dim, pair_dim | ConformerEmbedder |
| `PairformerArgs` | atom_dim, pair_dim, num_blocks, num_heads, head_dim | PairFormer |
| `InstanceSegArgs` | channels=64, attention_levels, num_selected_points=8192 | InstanceSegModule |
| `DiffusionArgs` | sigma_min/max, sigma_data=16.0, rho=7.0 | AtomDiffusion |
| `DiffusionPredictArgs` | multiplicity=4, num_sampling_steps=20 | Emap2lig.predict_step |
| `AuxiliaryArgs` | num_bins=20, num_elements=6, num_bond_types=5, num_chirality_types=7, num_ring_sizes=4 | AuxiliaryModule |

## WHERE TO LOOK

| Task | Location |
|------|----------|
| Add new prediction head | `modules/pairformer.py` → AuxiliaryModule |
| Modify diffusion schedule | `modules/diffusion.py` → AtomDiffusion |
| Change atom feature input | `modules/conf_embedder.py` → ConformerEmbedder.forward |
| Tune density processing | `modules/instance_seg.py` → InstanceSegModule |
| Change attention mechanism | `layers/attention.py` or `layers/triangular_attention.py` |
| Modify segmentation model | `seg/model.py` → MUNetRegSeg |

## WEIGHTS

Downloaded from HuggingFace `KiharaLab/Emap2lig` to `~/.emap2lig/models/`:
- Build model: `emap2lig-build-v0.0.1.safetensors`
- Find model (detection): `emap2lig-find-v0.0.1.safetensors`
- Fragment detection: `emap2lig-frag.safetensors`

Loaded with `strict=False` in `load_state_dict()`.

## ANTI-PATTERNS

- `strict=False` on state dict loading — hides structural drift between checkpoint and model
- Dict/DictConfig args coerced to dataclasses in `__init__` — fragile pattern from Hydra integration
