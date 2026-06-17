# Supported Platforms

Emap2lig local inference is accelerator-only. CPU inference is not supported.

| Platform | Local inference accelerator | Requirements |
|----------|-----------------------------|--------------|
| Linux | NVIDIA CUDA | CUDA 12/13 compatible driver; 8 GB+ VRAM recommended |
| macOS | Apple MPS | MPS-capable Mac; macOS 13.2+ |

The runtime selects the accelerator by platform:

- Linux uses CUDA and validates the `--gpu` CUDA device ID.
- macOS uses the single MPS device (`--gpu 0`).
- Other platforms fail fast with an unsupported-platform error.

## macOS notes

macOS inference uses PyTorch's MPS backend. The pipeline relies on 3D
convolution and trilinear 3D upsampling on MPS, so Emap2lig requires
`torch>=2.8.0` and macOS 13.2 or newer.

CUDA-only optimized kernels, such as cuequivariance triangle kernels and Flash
Attention paths, are automatically bypassed on MPS in favor of the standard
PyTorch implementations.

## No supported accelerator?

Use the [KiharaLab web server](web-server.md), which does not require local GPU
hardware or installation.
