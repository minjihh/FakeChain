# FakeChain: Exposing Shallow Cues in Multi-Step Deepfake Detection

[![CIKM 2025 - Accepted](https://img.shields.io/badge/CIKM%202025-Accepted-brightgreen.svg)](https://arxiv.org/abs/2509.16602)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC--BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

We are providing samples of **multi-step** manipulations under various settings.

![FakeChain Overview](fig/cikm_thumb.PNG)


## 🧩 Manipulation Depth & Composition

We organize the dataset by manipulation depth:

- `1Step/`: Single-generator manipulations
  - `FF/`   – FaceSwap (FaceFusion)
  - `SG3/`  – GAN-based (StyleGAN3)
  - `SS/`   – GAN-based (StyleSwin)
  - `SD3/`  – Diffusion-based (Stable Diffusion 3)
  - `SDXL/` – Diffusion-based (Stable Diffusion XL)

- `2Step/`: Two-step compositional manipulations  
  Each subfolder corresponds to an ordered pair of **different generators**.  
  Pairs such as `FF_SG3`, `SG3_SD3`, or `SDXL_FF` are included.  
  However, same-generator pairs (e.g., `FF_FF`, `SG3_SG3`, `SS_SS`, `SD3_SD3`, `SDXL_SDXL`) are **not** used.

  Note: Both cross-GAN directions (e.g., `SG3_SS` and `SS_SG3`) **do** exist, but **GAN→same GAN** (e.g., `SG3_SG3`, `SS_SS`) does not.

- `3Step/`: Three-step compositional manipulations  
  Only combinations that contain **one method from each family** are included:  
  - FaceSwap: `FF`
  - GAN: `SG3` or `SS`
  - Diffusion: `SD3` or `SDXL`

  Therefore, **no two-step repetition of the same family occurs in the chain**.  
  For example, `FF_SG3_SD3`, `SDXL_SS_FF`, etc., are included, whereas  
  combinations containing two GANs (e.g., `FF_SG3_SS`, `SG3_SS_SD3`) or two Diffusion models (e.g., `FF_SD3_SDXL`) are **not** included.


---

## 📥 Dataset Access

The FakeChain dataset is hosted on **Harvard Dataverse**.  
Due to identity and licensing considerations, the dataset is **restricted** and not publicly downloadable.

To request access, please visit the link below and click **“Request Access”** on the right side of the file list:

🔗 https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ATZAEJ

Once a request is submitted, I will manually review and grant access as soon as possible.

---

## 📌 Usage Policy

This dataset is provided **for research and educational purposes only**.  
Any form of **redistribution, commercial use, or public demonstration** is strictly prohibited.

License: **CC BY-NC 4.0**

---

## ⚠️ HuggingFace Release (In Progress)

A full mirrored version of FakeChain on **HuggingFace** is planned for improved accessibility.  
However, due to ongoing **storage and hosting constraints**, the upload is still in progress.

The repository will be updated as soon as the HuggingFace release becomes available.

---

## 📨 Contact

If you have questions regarding dataset access, feel free to contact:

📩 **minji.h0224@g.skku.edu**

 
