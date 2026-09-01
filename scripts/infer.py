#!/usr/bin/env python3
# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run MMDiff inference through the native Diffusers custom pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from diffusers import DiffusionPipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate OPT/SAR/IR images with MMDiffPipeline.")
    parser.add_argument(
        "--model_path",
        default="./models/MMDiff",
        help="Packaged Diffusers directory produced by scripts/convert_mmdiff_to_diffusers.py.",
    )
    parser.add_argument(
        "--prompt",
        default="There is a ship in the blue water on the shore.",
        help="Text prompt shared by all modalities.",
    )
    parser.add_argument("--negative_prompt", default="", help="Optional negative prompt.")
    parser.add_argument("--scene", default="ship", help="LoRA scene name (ship, beach, river, ...).")
    parser.add_argument(
        "--modalities",
        default="opt,sar,ir",
        help="Comma-separated subset of opt,sar,ir or 'all'.",
    )
    parser.add_argument("--output_dir", default="result", help="Directory used to write PNG files.")
    parser.add_argument("--name", default="ship", help="Filename stem for the saved images.")
    parser.add_argument("--seed", type=int, default=2026, help="Generator seed.")
    parser.add_argument("--device", default="cuda:0", help="Torch device, e.g. cuda:0 or cpu.")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--resnet_layer", type=int, default=2, help="0-based up-block ResNet index to inject.")
    parser.add_argument("--resnet_time", type=float, default=1.0)
    parser.add_argument(
        "--attn_layers",
        default="1,2,3,4,5,6,7,8,9",
        help="Comma-separated 1-based self-attention layers to transfer.",
    )
    parser.add_argument("--sar_lora_path", default=None, help="Optional explicit SAR LoRA directory.")
    parser.add_argument("--ir_lora_path", default=None, help="Optional explicit IR LoRA directory.")
    return parser.parse_args()


def parse_modalities(raw: str) -> str | list[str]:
    value = raw.strip()
    if value.lower() == "all":
        return "all"
    return [item.strip() for item in value.split(",") if item.strip()]


def load_pipeline(model_path: Path, dtype: torch.dtype):
    pipeline_file = model_path / "pipeline.py"
    kwargs = {
        "torch_dtype": dtype,
        "safety_checker": None,
        "requires_safety_checker": False,
    }
    if pipeline_file.exists():
        kwargs.update(
            custom_pipeline=str(pipeline_file),
            trust_remote_code=True,
        )
        return DiffusionPipeline.from_pretrained(str(model_path), **kwargs)

    from pipeline import MMDiffPipeline

    return MMDiffPipeline.from_pretrained(str(model_path), **kwargs)


def save_images(images, output_dir: Path, modality: str, name: str) -> list[Path]:
    modality_dir = output_dir / modality
    modality_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for index, image in enumerate(images):
        filename = f"{name}.png" if index == 0 else f"{name}_{index}.png"
        path = modality_dir / filename
        image.save(path)
        written.append(path)
    return written


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_path}. "
            "Run scripts/convert_mmdiff_to_diffusers.py first."
        )

    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    pipe = load_pipeline(model_path, dtype=dtype)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=False)

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    modalities = parse_modalities(args.modalities)
    attn_layers = [int(item) for item in args.attn_layers.split(",") if item.strip()]

    output = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt or None,
        scene=args.scene,
        modalities=modalities,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        attn_layers=attn_layers,
        resnet_layers=[args.resnet_layer],
        resnet_time=args.resnet_time,
        sar_lora_path=args.sar_lora_path,
        ir_lora_path=args.ir_lora_path,
        output_type="pil",
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    written = []
    for modality in ("opt", "sar", "ir"):
        images = getattr(output, modality)
        if images:
            written.extend(save_images(images, output_dir, modality, args.name))

    print(f"Saved {len(written)} image(s) under {output_dir}")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
