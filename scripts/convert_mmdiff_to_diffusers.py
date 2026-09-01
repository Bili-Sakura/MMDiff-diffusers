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

"""Package original MMDiff weights into a Diffusers custom-pipeline directory."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

DEFAULT_HUB_ID = "XinRan-Tang/MM-Diff"
OPT_COMPONENTS = ("unet", "vae", "text_encoder", "tokenizer", "scheduler")
LORA_WEIGHT_NAMES = (
    "pytorch_lora_weights.safetensors",
    "pytorch_lora_weights.bin",
    "adapter_model.safetensors",
    "adapter_model.bin",
)
SNAPSHOT_ALLOW_PATTERNS = (
    "mmdiff-opt/model_index.json",
    "mmdiff-opt/unet/*",
    "mmdiff-opt/vae/*",
    "mmdiff-opt/text_encoder/*",
    "mmdiff-opt/tokenizer/*",
    "mmdiff-opt/scheduler/*",
    "mmdiff-sar/*/pytorch_lora_weights.safetensors",
    "mmdiff-ir/*/pytorch_lora_weights.safetensors",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert XinRan-Tang/MM-Diff (or a local copy) into a Diffusers custom-pipeline "
            "repo that can be loaded with DiffusionPipeline.from_pretrained(..., custom_pipeline=...)."
        )
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_HUB_ID,
        help="Local MMDiff weight root or Hugging Face repo id. Defaults to XinRan-Tang/MM-Diff.",
    )
    parser.add_argument(
        "--dump_path",
        required=True,
        help="Output directory for the packaged Diffusers pipeline.",
    )
    parser.add_argument(
        "--opt_subdir",
        default="mmdiff-opt",
        help="Subdirectory of --source that holds the optical Stable Diffusion pipeline.",
    )
    parser.add_argument(
        "--sar_subdir",
        default="mmdiff-sar",
        help="Subdirectory of --source that holds SAR LoRA adapters.",
    )
    parser.add_argument(
        "--ir_subdir",
        default="mmdiff-ir",
        help="Subdirectory of --source that holds IR LoRA adapters.",
    )
    parser.add_argument(
        "--opt_checkpoint",
        default=None,
        help="Optional training checkpoint name under the OPT folder, e.g. checkpoint-15000.",
    )
    parser.add_argument(
        "--from_ckpt",
        default=None,
        help="Optional original Stable Diffusion .ckpt/.safetensors to bootstrap missing OPT components.",
    )
    parser.add_argument(
        "--scenes",
        default=None,
        help="Comma-separated scene names to package. Default: every detected scene.",
    )
    parser.add_argument(
        "--lora_root",
        default="loras",
        help="Name of the LoRA folder written inside --dump_path.",
    )
    parser.add_argument(
        "--scheduler_class",
        default="DDPMScheduler",
        help="Scheduler class name written to scheduler/scheduler_config.json. Original sampling used DDPMScheduler.",
    )
    parser.add_argument(
        "--skip_pipeline_copy",
        action="store_true",
        help="Do not copy pipeline.py into the output directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace --dump_path if it already exists.",
    )
    return parser.parse_args()


def _copytree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Required path does not exist: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def resolve_source(source: str) -> Path:
    path = Path(source).expanduser()
    if path.exists():
        return path.resolve()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise ImportError("Install huggingface_hub to download --source repo ids.") from error

    print(f"Downloading {source} (inference weights only; training checkpoints are skipped)...")
    cached = snapshot_download(repo_id=source, allow_patterns=list(SNAPSHOT_ALLOW_PATTERNS))
    return Path(cached)


def detect_scenes(lora_root: Path, prefix: str) -> list[str]:
    if not lora_root.is_dir():
        return []
    scenes: list[str] = []
    for child in sorted(lora_root.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        if name.startswith(prefix):
            scenes.append(name[len(prefix) :])
        elif (child / "pytorch_lora_weights.safetensors").exists():
            scenes.append(name)
    return scenes


def find_lora_dir(lora_root: Path, prefix: str, scene: str) -> Path | None:
    candidates = [
        lora_root / f"{prefix}{scene}",
        lora_root / scene,
        lora_root / f"sd-{prefix.rstrip('-')}{scene}",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def copy_lora(src_dir: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = None
    for filename in LORA_WEIGHT_NAMES:
        src = src_dir / filename
        if src.exists():
            shutil.copy2(src, dst_dir / "pytorch_lora_weights.safetensors" if filename.endswith(".safetensors") else dst_dir / filename)
            copied = dst_dir / ("pytorch_lora_weights.safetensors" if filename.endswith(".safetensors") else filename)
            break
    if copied is None:
        raise FileNotFoundError(f"No LoRA weights found under {src_dir}")
    adapter_config = src_dir / "adapter_config.json"
    if adapter_config.exists():
        shutil.copy2(adapter_config, dst_dir / "adapter_config.json")
    return copied


def copy_opt_components(opt_root: Path, dump_path: Path, opt_checkpoint: str | None) -> None:
    for component in OPT_COMPONENTS:
        src = opt_root / component
        if component == "unet" and opt_checkpoint:
            ckpt_unet = opt_root / opt_checkpoint / "unet"
            if ckpt_unet.exists():
                src = ckpt_unet
        if not src.exists():
            raise FileNotFoundError(f"Missing OPT component '{component}' at {src}")
        _copytree(src, dump_path / component)


def maybe_bootstrap_from_ckpt(from_ckpt: str, dump_path: Path) -> None:
    from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt

    print(f"Converting original Stable Diffusion checkpoint {from_ckpt} ...")
    pipe = download_from_original_stable_diffusion_ckpt(
        checkpoint_path_or_dict=from_ckpt,
        from_safetensors=str(from_ckpt).endswith(".safetensors"),
        scheduler_type="ddpm",
    )
    pipe.save_pretrained(dump_path, safe_serialization=True)


def rewrite_scheduler(dump_path: Path, scheduler_class: str) -> None:
    config_path = dump_path / "scheduler" / "scheduler_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["_class_name"] = scheduler_class
    # Keep only scheduler fields; drop unused trained-beta blobs if present.
    allowed = {
        "_class_name",
        "_diffusers_version",
        "beta_end",
        "beta_schedule",
        "beta_start",
        "clip_sample",
        "num_train_timesteps",
        "prediction_type",
        "set_alpha_to_one",
        "skip_prk_steps",
        "steps_offset",
        "timestep_spacing",
        "trained_betas",
        "variance_type",
        "rescale_betas_zero_snr",
    }
    config = {key: value for key, value in config.items() if key in allowed}
    _write_json(config_path, config)


def write_model_index(dump_path: Path, lora_root: str, scheduler_class: str) -> None:
    payload = {
        "_class_name": "MMDiffPipeline",
        "_diffusers_version": "0.32.0",
        "feature_extractor": [None, None],
        "image_encoder": [None, None],
        "lora_root": lora_root,
        "requires_safety_checker": False,
        "safety_checker": [None, None],
        "scheduler": ["diffusers", scheduler_class],
        "text_encoder": ["transformers", "CLIPTextModel"],
        "tokenizer": ["transformers", "CLIPTokenizer"],
        "unet": ["diffusers", "UNet2DConditionModel"],
        "vae": ["diffusers", "AutoencoderKL"],
    }
    _write_json(dump_path / "model_index.json", payload)


def copy_pipeline_module(dump_path: Path) -> None:
    src = Path(__file__).resolve().parents[1] / "pipeline.py"
    if not src.exists():
        raise FileNotFoundError(f"pipeline.py not found next to the repository root: {src}")
    shutil.copy2(src, dump_path / "pipeline.py")


def filter_scenes(detected: Iterable[str], requested: str | None) -> list[str]:
    scenes = list(dict.fromkeys(detected))
    if requested:
        wanted = [item.strip() for item in requested.split(",") if item.strip()]
        missing = [item for item in wanted if item not in scenes]
        if missing:
            raise ValueError(f"Requested scenes {missing} were not found. Available: {scenes}")
        scenes = wanted
    return scenes


def convert(args: argparse.Namespace) -> Path:
    dump_path = Path(args.dump_path).expanduser().resolve()
    if dump_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"{dump_path} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(dump_path)
    dump_path.mkdir(parents=True, exist_ok=True)

    if args.from_ckpt:
        maybe_bootstrap_from_ckpt(args.from_ckpt, dump_path)

    source = resolve_source(args.source)
    opt_root = source / args.opt_subdir if (source / args.opt_subdir).exists() else source
    if not (opt_root / "unet").exists() and not args.from_ckpt:
        raise FileNotFoundError(
            f"Could not find an OPT Diffusers pipeline under {opt_root}. "
            "Pass --from_ckpt to bootstrap from an original Stable Diffusion checkpoint."
        )

    if (opt_root / "unet").exists():
        copy_opt_components(opt_root, dump_path, args.opt_checkpoint)

    sar_root = source / args.sar_subdir
    ir_root = source / args.ir_subdir
    scenes = filter_scenes(
        detect_scenes(sar_root, "sd-sar-lora-") or detect_scenes(ir_root, "sd-ir-lora-"),
        args.scenes,
    )

    packaged = []
    for scene in scenes:
        for modality, root, prefix in (
            ("sar", sar_root, "sd-sar-lora-"),
            ("ir", ir_root, "sd-ir-lora-"),
        ):
            src_dir = find_lora_dir(root, prefix, scene)
            if src_dir is None:
                print(f"Warning: no {modality.upper()} LoRA for scene '{scene}'")
                continue
            dst = dump_path / args.lora_root / modality / scene
            copy_lora(src_dir, dst)
            packaged.append(f"{modality}/{scene}")

    rewrite_scheduler(dump_path, args.scheduler_class)
    write_model_index(dump_path, args.lora_root, args.scheduler_class)
    if not args.skip_pipeline_copy:
        copy_pipeline_module(dump_path)

    print(f"Wrote Diffusers MMDiff pipeline to {dump_path}")
    if packaged:
        print("Packaged LoRA adapters: " + ", ".join(packaged))
    else:
        print("No SAR/IR LoRA adapters were packaged. OPT-only generation will still work.")
    return dump_path


def main() -> None:
    convert(parse_args())


if __name__ == "__main__":
    main()
