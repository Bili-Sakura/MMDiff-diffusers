"""Helpers that build a tiny MMDiff pipeline without pretrained weights."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel

from pipeline import MMDiffPipeline


def write_tiny_tokenizer(path: Path, vocab_size: int = 64) -> CLIPTokenizer:
    path.mkdir(parents=True, exist_ok=True)
    vocab = {
        "<|startoftext|>": 0,
        "<|endoftext|>": 1,
        "!": 2,
        "a</w>": 3,
        "ship</w>": 4,
        "there</w>": 5,
        "is</w>": 6,
        "water</w>": 7,
    }
    next_id = max(vocab.values()) + 1
    for index in range(next_id, vocab_size):
        vocab[f"tok{index}</w>"] = index
    (path / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    (path / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    (path / "special_tokens_map.json").write_text(
        json.dumps(
            {
                "bos_token": "<|startoftext|>",
                "eos_token": "<|endoftext|>",
                "unk_token": "<|endoftext|>",
                "pad_token": "<|endoftext|>",
            }
        ),
        encoding="utf-8",
    )
    (path / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "CLIPTokenizer", "model_max_length": 77}),
        encoding="utf-8",
    )
    return CLIPTokenizer.from_pretrained(str(path))


def make_tiny_components(tokenizer_dir: Path):
    torch.manual_seed(0)
    unet = UNet2DConditionModel(
        sample_size=8,
        in_channels=4,
        out_channels=4,
        layers_per_block=2,
        block_out_channels=(32, 64),
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim=32,
        attention_head_dim=8,
        norm_num_groups=8,
    )
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(32, 32),
        latent_channels=4,
        norm_num_groups=8,
        sample_size=16,
    )
    text_encoder = CLIPTextModel(
        CLIPTextConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=77,
            bos_token_id=0,
            eos_token_id=1,
            pad_token_id=1,
        )
    )
    tokenizer = write_tiny_tokenizer(tokenizer_dir)
    scheduler = DDPMScheduler(
        num_train_timesteps=10,
        beta_start=0.00085,
        beta_end=0.012,
        clip_sample=False,
        steps_offset=1,
    )
    return {
        "unet": unet,
        "vae": vae,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "scheduler": scheduler,
    }


def make_tiny_pipeline(tokenizer_dir: Path) -> MMDiffPipeline:
    components = make_tiny_components(tokenizer_dir)
    return MMDiffPipeline(
        **components,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )


def make_dummy_source_tree(root: Path, scenes: tuple[str, ...] = ("ship", "beach")) -> Path:
    """Create a fake XinRan-Tang/MM-Diff layout for conversion tests."""

    opt = root / "mmdiff-opt"
    for component in ("unet", "vae", "text_encoder", "tokenizer", "scheduler"):
        (opt / component).mkdir(parents=True)
        (opt / component / "config.json").write_text("{}", encoding="utf-8")
        (opt / component / "dummy.bin").write_bytes(b"weights")
    (opt / "model_index.json").write_text(
        json.dumps({"_class_name": "StableDiffusionPipeline", "unet": ["diffusers", "UNet2DConditionModel"]}),
        encoding="utf-8",
    )
    (opt / "scheduler" / "scheduler_config.json").write_text(
        json.dumps(
            {
                "_class_name": "PNDMScheduler",
                "beta_start": 0.00085,
                "beta_end": 0.012,
                "beta_schedule": "scaled_linear",
                "num_train_timesteps": 1000,
                "prediction_type": "epsilon",
                "clip_sample": False,
                "steps_offset": 1,
            }
        ),
        encoding="utf-8",
    )
    (opt / "checkpoint-15000" / "unet").mkdir(parents=True)
    (opt / "checkpoint-15000" / "unet" / "config.json").write_text('{"from":"ckpt"}', encoding="utf-8")

    for scene in scenes:
        for modality, prefix in (("sar", "sd-sar-lora-"), ("ir", "sd-ir-lora-")):
            lora_dir = root / f"mmdiff-{modality}" / f"{prefix}{scene}"
            lora_dir.mkdir(parents=True)
            (lora_dir / "pytorch_lora_weights.safetensors").write_bytes(b"lora")
            (lora_dir / "checkpoint-15000").mkdir()
            (lora_dir / "checkpoint-15000" / "model.safetensors").write_bytes(b"skip-me")
    return root
