# FakeChain: Exposing Shallow Cues in Multi-Step Deepfake Detection

[![CIKM 2025 - Accepted](https://img.shields.io/badge/CIKM%202025-Accepted-brightgreen.svg)](https://arxiv.org/abs/2509.16602)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC--BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

We are providing samples of **multi-step** manipulations under various settings.

![FakeChain Overview](fig/cikm_thumb.PNG)


### Manipulation Depth & Composition

We organize the dataset by manipulation depth:

- `1Step/`: Single-generator manipulations
  - `FF/`   – FaceSwap (FaceFusion)
  - `SG3/`  – GAN-based (StyleGAN3)
  - `SS/`   – GAN-based (StyleSwin)
  - `SD3/`  – Diffusion-based (Stable Diffusion 3)
  - `SDXL/` – Diffusion-based (Stable Diffusion XL)

- `2Step/`: Two-step compositional manipulations  
  Each subfolder corresponds to an ordered pair of distinct generators, e.g.,
  `FF_SG3` (FaceFusion → StyleGAN3), `SG3_SD3` (StyleGAN3 → Stable Diffusion 3), etc.  
  Homogeneous pairs (same generator twice) are not used.

- `3Step/`: Three-step compositional manipulations  
  Homogeneous combinations are excluded to prioritize realistic and diverse compositional scenarios.  
  Each 3-step chain selects exactly one method from each family:
  - FaceSwap: `FF`
  - GAN: `SG3` or `SS`
  - Diffusion: `SD3` or `SDXL`


 
