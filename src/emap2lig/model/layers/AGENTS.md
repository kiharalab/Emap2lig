# Model Layers Agent

**Last Reviewed:** 2026-02-12
**Scope Note:** No functional changes in this module from the current web UI update set.

## OVERVIEW
High-fidelity structural layers combining AF2-style triangular logic with 3D spatial encodings and hardware-optimized attention.

## STRUCTURE

```
layers/
├── attention.py            # Multi-head attention with pairwise bias
├── triangular_attention.py # AlphaFold2 axial attention
├── triangle_mult.py        # Triangular multiplicative updates
├── selected_attention.py   # Cross-attention on pre-selected point features
├── instance.py             # Instance segmentation blocks
├── outer_product_mean.py   # Pairwise interaction projection
├── decoder.py              # Coordinate regression head
├── positional_encoding.py  # 3D sinusoidal spatial encodings
├── fourier.py              # Temporal signal embeddings
├── transition.py           # Gated feature transformation
├── primitives.py           # Base normalization and activation layers
├── dropout.py              # Pair-aware dropout mechanisms
└── __init__.py             # Module API and exports
```

## KEY COMPONENTS
- **attention.py**: Multi-head attention using pairwise structural bias projected from $c_z$. Uses `torch.compile` for memory efficiency.
- **triangular_attention.py**: Implementation of AlphaFold2 axial attention (Algorithms 13 & 14). Updates pair representations by attending along one axis with cross-axis bias.
- **triangle_mult.py**: Triangular multiplicative updates for pair representations (Incoming/Outgoing directions). Features optimized `cuequivariance` CUDA kernels.
- **selected_attention.py**: Cross-attention between atom queries and pre-selected point features from InstanceSegModule. Uses 3D positional encoding and FlashAttention for efficient point-to-atom conditioning.
- **instance.py**: Core instance segmentation blocks. Integrates 3D convolutions with volume-to-atom cross-attention and relative positional encodings.
- **outer_product_mean.py**: Computes pairwise interactions from sequence representations. Uses gated outer products mean-pooled across the sequence dimension.
- **decoder.py**: Terminal atom coordinate decoder. Employs LayerNorm and linear projection to map latent atom features to absolute 3D Cartesian coordinates.
- **positional_encoding.py**: 3D sinusoidal encoding for point clouds. Maps continuous XYZ coordinates into frequency-based feature vectors for spatial awareness.
- **fourier.py**: Fourier feature embedding layer for continuous time values. Generates periodic embeddings for diffusion timesteps via fixed random projections.
- **transition.py**: Gated transition layer (MLP). Utilizes dual linear projections with SiLU gating for non-linear feature transformation in structural trunks.
- **primitives.py**: Primitive building blocks including `LayerNorm` (BF16-optimized), `SwiGLU` activation, and standard `MLP` architectures.
- **dropout.py**: Pair-representation-specific dropout. Implements rowwise and columnwise masking to maintain structural consistency in 2D pair features.

## PATTERNS
- **Geometric Priors**: Structural constraints are enforced via pairwise biases, triangular consistency logic, and 3D sinusoidal spatial signal injection.
- **Compute Optimization**: Peak performance achieved through `cuequivariance` kernels, FlashAttention, and JIT-compiled modules.
- **Mixed Precision Guardrails**: Custom normalization modules force FP32 accumulation during reduction to ensure numerical stability for BF16/FP16 training.
- **High-Fidelity Gating**: Pervasive use of Gated Linear Units (SwiGLU, SiLU-gated MLP) for selective and context-aware feature propagation.
- **Grid-to-Point Bridge**: Top-k point selection from 3D voxel grids based on instance probabilities, enabling efficient cross-attention between density features and atom representations.
- **Implementation Flexibility**: Layers provide fallback PyTorch implementations alongside hardware-specific kernels for portability across platforms.
