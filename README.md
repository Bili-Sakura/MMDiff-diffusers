<div align="center">

<img src="assets/logo.png" alt="MMDiff logo" width="580" />

# MMDiff: Multi-modal Remote Sensing Image Generation via Cross-Modality Spatial Feature Transfer

<div  align="center" style="margin-top:10px;"> 
<img src="assets/ribbon.svg" alt="decorative dashed ribbon" width="760" />
<p><span style="font-size: 26px;"><strong>ISPRS 2026 &#128293;</strong></span></p>
<div align="center">

**Haojun Tang**<sup>1</sup> · **Wenda Zhao**<sup>1,*</sup> · **Hengshuai Cui**<sup>1</sup> · **Haipeng Wang**<sup>2</sup>

<sup>1</sup> Dalian University of Technology  
<sup>2</sup> Unit 92728 of PLA

</div>

<div align="center">

[![Website](https://img.shields.io/badge/Homepage-Website-87CEEB)](https://xinr-tang.github.io/MMDiff-homepage/)
[![ISPRS](https://img.shields.io/badge/ISPRS-Paper-2563EB)](https://www.sciencedirect.com/science/article/pii/S0924271626004089?dgcid=author)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-F9D371)](https://huggingface.co/XinRan-Tang/MM-Diff)
[![Dataset](https://img.shields.io/badge/Dataset-Access-green)](https://huggingface.co/datasets/XinRan-Tang/Optical-SAR-Infrared)

</div>

<sup>*</sup> **Corresponding author:**

</div>


## Abstract

<div align="justify">

Collecting spatially consistent multi-modal remote sensing (MMRS) images remains challenging due to different sensors vary in the imaging principles and acquisition times. This hinders the development of data-driven MMRS technologies, which rely on large-scale training samples. This paper proposes MMDiff, the first text-driven diffusion framework explicitly designed for jointly generating structurally consistent optical (OPT), synthetic aperture radar (SAR), and infrared (IR) remote sensing images from a single text prompt via cross-modality spatial feature transfer. MMDiff first trains the OPT branch with paired optical image-text data to capture rich semantic content, and then trains the SAR/IR branches with simple modality-specific text templates to learn the corresponding style attributes, without relying on complex linguistic descriptions. Specifically, we introduce a LoRA-based modality translation adaptation mechanism to translate the style attributes of optical spatial representations to SAR and IR style attributes while preserving the underlying semantic content. The translated representations are then transferred into the SAR and IR generation branches through the proposed spatial feature transfer mechanism, enabling rich spatial details in the generated SAR/IR images while maintaining cross-modal spatial consistency. Extensive experiments demonstrate that MMDiff achieves superior image quality in terms of modality similarity and semantic consistency compared to the state-of-the-art methods. Furthermore, MMDiff benefits downstream data-driven MMRS applications, e.g., multi-modal image fusion and object classification.

</div>

<p align="center"><img src="assets/show.png" alt="Generated multi-modal remote sensing images" width="900" /></p>
<p align="center" style="margin-top: -10px;"><span style="color: gray; font-size: 14px;">From top to bottom: optical (OPT), synthetic aperture radar (SAR), and infrared (IR) images.</span></p>

<div align="left">

## 📢 &nbsp; Latest Updates

- **2026-09-01** — Inference now uses a native Diffusers custom pipeline (`pipeline.py`) plus a weight conversion script.
- **2026-08-25** — Training code will be released soon.
- **2026-08-25** — Dataset and model are available on Hugging Face 🎊 ！
- **2026-08-25** — Sampling code is now available ✨.
- **2026-08-15** — Our paper has been accepted by **ISPRS 2026**  🎉 🎉 🎉 !!! 


## 📊 &nbsp; Main Results

### 1️⃣ &nbsp; Text-to-spatially-consistent Optical, Synthetic Aperture Radar, and Infrared Generation

<p align="center"><img src="assets/results2.png" alt="Text-to-spatially-aligned OPT, SAR, and IR generation" width="1000" /></p>

### 2️⃣ &nbsp; Comparison with Existing Methods

<p align="center"><img src="assets/comparison.png" alt="Comparison with existing methods" width="1000" /></p>

### 3️⃣ &nbsp; Downstream Tasks

<p align="center"><img src="assets/downstream.png" alt="Downstream tasks" width="1000" /></p>

## 🚀 Quick Start

### 🛠️ Installation

```bash
git clone https://github.com/Bili-Sakura/MMDiff-diffusers
cd MMDiff-diffusers

conda create -n mmdiff python=3.10 -y
conda activate mmdiff
pip install -r requirements.txt
```

### 📦 Convert official weights

The published [XinRan-Tang/MM-Diff](https://huggingface.co/XinRan-Tang/MM-Diff) repo already stores a Stable Diffusion OPT pipeline plus SAR/IR LoRA files. Package it into a native Diffusers custom-pipeline directory (this skips training checkpoints):

```bash
python scripts/convert_mmdiff_to_diffusers.py \
  --source XinRan-Tang/MM-Diff \
  --dump_path ./models/MMDiff
```

If you already downloaded the original tree locally:

```bash
python scripts/convert_mmdiff_to_diffusers.py \
  --source /path/to/XinRan-Tang/MM-Diff \
  --dump_path ./models/MMDiff
```

The packaged layout is:

```text
models/MMDiff/
├── pipeline.py
├── model_index.json
├── unet/  vae/  text_encoder/  tokenizer/  scheduler/
└── loras/
    ├── sar/<scene>/pytorch_lora_weights.safetensors
    └── ir/<scene>/pytorch_lora_weights.safetensors
```

Optional `--from_ckpt` bootstraps missing OPT components from an original Stable Diffusion `.ckpt` / `.safetensors` file. Use `--opt_checkpoint checkpoint-15000` to take the OPT UNet from a training snapshot instead of the final `mmdiff-opt/unet` folder.

### 🛠️ Inference

Load the custom pipeline with the standard Diffusers one-stop API. Spatial feature transfer now stays in memory inside `MMDiffPipeline` (no disk dumps under `features/`).

```python
from pathlib import Path
import torch
from diffusers import DiffusionPipeline

model_dir = Path("./models/MMDiff")
pipe = DiffusionPipeline.from_pretrained(
    str(model_dir),
    local_files_only=True,
    custom_pipeline=str(model_dir / "pipeline.py"),
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
pipe = pipe.to("cuda")

generator = torch.Generator(device="cpu").manual_seed(2026)
output = pipe(
    "There is a ship in the blue water on the shore.",
    scene="ship",
    height=256,
    width=256,
    num_inference_steps=50,
    generator=generator,
)
output.opt[0].save("result/opt/ship.png")
output.sar[0].save("result/sar/ship.png")
output.ir[0].save("result/ir/ship.png")
```

Or use the CLI / shell wrapper:

```bash
python scripts/infer.py \
  --model_path ./models/MMDiff \
  --prompt "There is a ship in the blue water on the shore." \
  --scene ship \
  --device cuda:0

# equivalent
bash scripts/sampling.sh
```

Generated OPT, SAR, and IR images are saved under `result/opt/`, `result/sar/`, and `result/ir/`. Scene LoRAs shipped by the official checkpoint include `beach`, `bridge`, `desert`, `farmland`, `lake`, `mountain`, `residential`, `river`, and `ship`.

Remote Hub ids use the same loading pattern after you push a packaged `UserID/mmdiff-diffusers` repo.

</div>

