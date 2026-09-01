from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffusers import DDIMScheduler, DDPMScheduler

from pipeline import (
    DEFAULT_RESOLUTION,
    MMDiffPipeline,
    SpatialFeatureStore,
    collect_up_resnets,
    collect_up_self_attentions,
    normalize_modalities,
)
from tests.tiny_helpers import make_tiny_pipeline


def test_normalize_modalities_orders_and_rejects_unknown() -> None:
    assert normalize_modalities("all") == ["opt", "sar", "ir"]
    assert normalize_modalities(["ir", "opt"]) == ["opt", "ir"]
    with pytest.raises(ValueError, match="Unsupported modalities"):
        normalize_modalities(["opt", "lidar"])
    with pytest.raises(ValueError, match="At least one"):
        normalize_modalities([])


def test_feature_store_roundtrip_and_injection_gates() -> None:
    store = SpatialFeatureStore()
    store.attn_layers = {1}
    store.resnet_layers = {2}
    store.mode = "save"
    store.set_timestep(torch.tensor(981))
    query = torch.randn(2, 4, 8)
    residual = torch.randn(2, 32, 8, 8)
    store.save_attn(1, query)
    store.save_resnet(2, residual)
    store.save_attn(9, query)
    store.save_resnet(0, residual)

    assert 1 in store.attn[981]
    assert 9 not in store.attn[981]
    assert 2 in store.resnet[981]
    assert 0 not in store.resnet[981]

    exported = store.to_state()
    restored = SpatialFeatureStore()
    restored.load_state(exported)
    restored.mode = "inject"
    restored.inject_attn_timesteps = {981}
    restored.inject_resnet_timesteps = {981}
    restored.set_timestep(981)
    assert torch.equal(restored.get_attn(1), exported["attn"][981][1])
    assert torch.equal(restored.get_resnet(2), exported["resnet"][981][2])

    restored.set_timestep(1)
    assert restored.get_attn(1) is None


def test_collect_up_modules_on_tiny_unet(tmp_path: Path) -> None:
    pipe = make_tiny_pipeline(tmp_path / "tok")
    attns = collect_up_self_attentions(pipe.unet)
    resnets = collect_up_resnets(pipe.unet)
    assert attns
    assert resnets
    assert all(module.to_q.in_features == module.to_k.in_features for module in attns)


def test_check_inputs_rejects_bad_spatial_features(tmp_path: Path) -> None:
    pipe = make_tiny_pipeline(tmp_path / "tok")
    with pytest.raises(TypeError, match="spatial_features"):
        pipe.check_inputs(
            prompt="a ship",
            height=16,
            width=16,
            callback_steps=None,
            spatial_features=["not-a-dict"],
        )


def test_opt_generation_is_reproducible(tmp_path: Path) -> None:
    pipe = make_tiny_pipeline(tmp_path / "tok")
    prompt_embeds = torch.randn(1, 8, 32)
    negative = torch.zeros_like(prompt_embeds)

    def _run(seed: int):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative,
            modalities=["opt"],
            height=16,
            width=16,
            num_inference_steps=2,
            guidance_scale=1.0,
            generator=generator,
            output_type="np",
            return_spatial_features=True,
        )

    first = _run(7)
    second = _run(7)
    third = _run(8)
    assert first.opt[0].shape[:2] == (16, 16)
    assert first.opt[0].shape[-1] == 3
    assert first.spatial_features is not None
    assert first.spatial_features["attn"] or first.spatial_features["resnet"]
    assert (first.opt[0] == second.opt[0]).all()
    assert not (first.opt[0] == third.opt[0]).all()


def test_scheduler_swap_and_multimodal_latent_output(tmp_path: Path) -> None:
    pipe = make_tiny_pipeline(tmp_path / "tok")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    assert isinstance(pipe.scheduler, DDIMScheduler)

    prompt_embeds = torch.randn(1, 8, 32)
    output = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=torch.zeros_like(prompt_embeds),
        modalities=["opt", "sar", "ir"],
        height=16,
        width=16,
        num_inference_steps=2,
        guidance_scale=1.0,
        generator=torch.Generator(device="cpu").manual_seed(0),
        output_type="latent",
    )
    assert output.opt.shape[1] == 4
    assert output.sar.shape == output.opt.shape
    assert output.ir.shape == output.opt.shape
    assert output.images is output.opt


def test_single_channel_decode(tmp_path: Path) -> None:
    pipe = make_tiny_pipeline(tmp_path / "tok")
    latents = torch.randn(1, 4, 2, 2, dtype=pipe.vae.dtype)
    rgb = pipe.decode_latents(latents, single_channel=False)
    gray = pipe.decode_latents(latents, single_channel=True)
    assert rgb.shape[1] == 3
    assert gray.shape[1] == 1


def test_default_resolution_and_save_load(tmp_path: Path) -> None:
    pipe = make_tiny_pipeline(tmp_path / "tok")
    assert DEFAULT_RESOLUTION == 256
    save_dir = tmp_path / "saved"
    pipe.save_pretrained(save_dir)
    assert (save_dir / "model_index.json").exists()
    assert (save_dir / "unet").exists()
    loaded = MMDiffPipeline.from_pretrained(str(save_dir), safety_checker=None, requires_safety_checker=False)
    assert loaded.config.lora_root == "loras"
    # Scheduler interchange after reload.
    loaded.scheduler = DDPMScheduler.from_config(loaded.scheduler.config)
    assert isinstance(loaded.scheduler, DDPMScheduler)


def test_prompt_encoding_and_grayscale_sar(tmp_path: Path) -> None:
    pipe = make_tiny_pipeline(tmp_path / "tok")
    output = pipe(
        prompt="a ship",
        modalities=["opt", "sar"],
        height=16,
        width=16,
        num_inference_steps=2,
        guidance_scale=7.5,
        generator=torch.Generator(device="cpu").manual_seed(1),
        output_type="pil",
    )
    assert output.opt[0].size == (16, 16)
    assert output.opt[0].mode in {"RGB", "RGBA"}
    assert output.sar[0].size == (16, 16)
    assert output.sar[0].mode == "L"


def test_custom_pipeline_from_pretrained(tmp_path: Path) -> None:
    import shutil

    from diffusers import DiffusionPipeline

    pipe = make_tiny_pipeline(tmp_path / "tok")
    save_dir = tmp_path / "custom"
    pipe.save_pretrained(save_dir)
    shutil.copy2(REPO_ROOT / "pipeline.py", save_dir / "pipeline.py")
    loaded = DiffusionPipeline.from_pretrained(
        str(save_dir),
        custom_pipeline=str(save_dir / "pipeline.py"),
        trust_remote_code=True,
        safety_checker=None,
        requires_safety_checker=False,
    )
    assert loaded.__class__.__name__ == "MMDiffPipeline"
    prompt_embeds = torch.randn(1, 8, 32)
    output = loaded(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=torch.zeros_like(prompt_embeds),
        modalities=["opt"],
        height=16,
        width=16,
        num_inference_steps=1,
        guidance_scale=1.0,
        output_type="latent",
    )
    assert output.opt.shape[1] == 4
