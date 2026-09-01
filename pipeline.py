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

"""Native Diffusers pipeline for MMDiff multi-modal remote-sensing generation."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from diffusers.image_processor import PipelineImageInput
from diffusers.loaders import FromSingleFileMixin, IPAdapterMixin, StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import Attention, AttnProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
    rescale_noise_cfg,
    retrieve_timesteps,
)
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import BaseOutput, logging, replace_example_docstring

logger = logging.get_logger(__name__)

DEFAULT_RESOLUTION = 256
DEFAULT_SCENE = "ship"
DEFAULT_ATTN_LAYERS = (1, 2, 3, 4, 5, 6, 7, 8, 9)
DEFAULT_RESNET_LAYERS = (2,)
SUPPORTED_MODALITIES = ("opt", "sar", "ir")
SUPPORTED_SCENES = (
    "beach",
    "bridge",
    "desert",
    "farmland",
    "lake",
    "mountain",
    "residential",
    "river",
    "ship",
)

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from pathlib import Path
        >>> import torch
        >>> from diffusers import DiffusionPipeline

        >>> model_dir = Path("/path/to/mmdiff-diffusers")
        >>> pipe = DiffusionPipeline.from_pretrained(
        ...     str(model_dir),
        ...     local_files_only=True,
        ...     custom_pipeline=str(model_dir / "pipeline.py"),
        ...     trust_remote_code=True,
        ...     torch_dtype=torch.bfloat16,
        ... )
        >>> pipe = pipe.to("cuda")

        >>> generator = torch.Generator(device="cpu").manual_seed(2026)
        >>> output = pipe(
        ...     "There is a ship in the blue water on the shore.",
        ...     scene="ship",
        ...     height=256,
        ...     width=256,
        ...     num_inference_steps=50,
        ...     generator=generator,
        ... )
        >>> output.opt[0].save("opt.png")
        >>> output.sar[0].save("sar.png")
        >>> output.ir[0].save("ir.png")

        >>> # Hugging Face Hub style model id: UserID/RepoID
        >>> # Example: "XinRan-Tang/MM-Diff" after conversion, or a packaged `mmdiff-diffusers` repo.
        ```
"""


def collect_up_self_attentions(unet: UNet2DConditionModel) -> list[Attention]:
    r"""
    Collect up-block self-attention modules in the same depth-first order used by the
    original MMDiff hook registration.

    Args:
        unet (`UNet2DConditionModel`):
            UNet whose `up_blocks` should be scanned.

    Returns:
        `list[Attention]`: Self-attention modules in 1-based transfer order.
    """

    modules: list[Attention] = []

    def _recurse(module: torch.nn.Module) -> None:
        if module.__class__.__name__ == "Attention":
            if module.to_q.in_features == module.to_k.in_features:
                modules.append(module)
            return
        for child in module.children():
            _recurse(child)

    _recurse(unet.up_blocks)
    return modules


def collect_up_resnets(unet: UNet2DConditionModel) -> list[torch.nn.Module]:
    r"""
    Collect up-block `ResnetBlock2D` modules in the original 0-based transfer order.

    Args:
        unet (`UNet2DConditionModel`):
            UNet whose `up_blocks` should be scanned.

    Returns:
        `list[torch.nn.Module]`: Residual blocks in injection-index order.
    """

    modules: list[torch.nn.Module] = []

    def _recurse(module: torch.nn.Module) -> None:
        if module.__class__.__name__ == "ResnetBlock2D":
            modules.append(module)
            return
        for child in module.children():
            _recurse(child)

    _recurse(unet.up_blocks)
    return modules


class SpatialFeatureStore:
    r"""
    In-memory store for OPT self-attention queries and up-block residual features.

    Features are keyed by integer scheduler timestep, then by layer index. This replaces
    the original disk dump under `features/visible_attn_maps` and `features/visible_resnet_maps`.
    """

    def __init__(self) -> None:
        self.attn: dict[int, dict[int, torch.Tensor]] = {}
        self.resnet: dict[int, dict[int, torch.Tensor]] = {}
        self.current_timestep: int | None = None
        self.mode: str = "off"
        self.attn_layers: set[int] = set(DEFAULT_ATTN_LAYERS)
        self.resnet_layers: set[int] = set(DEFAULT_RESNET_LAYERS)
        self.inject_attn_timesteps: set[int] | None = None
        self.inject_resnet_timesteps: set[int] | None = None

    def reset(self) -> None:
        r"""Clear captured features and timestep state without changing layer settings."""

        self.attn = {}
        self.resnet = {}
        self.current_timestep = None

    def set_timestep(self, timestep: int | torch.Tensor) -> None:
        r"""
        Record the scheduler timestep used by the current UNet forward.

        Args:
            timestep (`int` or `torch.Tensor`):
                Scalar diffusion timestep. Tensors are stored as `int`.
        """

        self.current_timestep = int(timestep)

    def _timestep_allowed(self, allowed: set[int] | None) -> bool:
        if self.current_timestep is None:
            return False
        if allowed is None:
            return True
        return self.current_timestep in allowed

    def save_attn(self, layer_idx: int, query: torch.Tensor) -> None:
        r"""
        Cache a self-attention query tensor for the current timestep.

        Args:
            layer_idx (`int`):
                1-based up-block self-attention index.
            query (`torch.Tensor`):
                Query tensor after `head_to_batch_dim`.
        """

        if self.mode != "save" or self.current_timestep is None:
            return
        if layer_idx not in self.attn_layers:
            return
        self.attn.setdefault(self.current_timestep, {})[layer_idx] = query.detach()

    def get_attn(self, layer_idx: int) -> torch.Tensor | None:
        r"""
        Return the cached query for the current timestep and layer, if injection is active.

        Args:
            layer_idx (`int`):
                1-based up-block self-attention index.

        Returns:
            `torch.Tensor` or `None`: Cached query, or `None` when injection does not apply.
        """

        if self.mode != "inject" or not self._timestep_allowed(self.inject_attn_timesteps):
            return None
        if layer_idx not in self.attn_layers:
            return None
        return self.attn.get(self.current_timestep, {}).get(layer_idx)

    def save_resnet(self, layer_idx: int, residual: torch.Tensor) -> None:
        r"""
        Cache an up-block residual tensor for the current timestep.

        Args:
            layer_idx (`int`):
                0-based up-block ResNet index.
            residual (`torch.Tensor`):
                `ResnetBlock2D` output.
        """

        if self.mode != "save" or self.current_timestep is None:
            return
        if layer_idx not in self.resnet_layers:
            return
        self.resnet.setdefault(self.current_timestep, {})[layer_idx] = residual.detach()

    def get_resnet(self, layer_idx: int) -> torch.Tensor | None:
        r"""
        Return the cached residual for the current timestep and layer, if injection is active.

        Args:
            layer_idx (`int`):
                0-based up-block ResNet index.

        Returns:
            `torch.Tensor` or `None`: Cached residual, or `None` when injection does not apply.
        """

        if self.mode != "inject" or not self._timestep_allowed(self.inject_resnet_timesteps):
            return None
        if layer_idx not in self.resnet_layers:
            return None
        return self.resnet.get(self.current_timestep, {}).get(layer_idx)

    def to_state(self) -> dict[str, Any]:
        r"""
        Export captured features for reuse in a later `__call__`.

        Returns:
            `dict`: Detached CPU tensors plus layer configuration.
        """

        def _cpu(store: dict[int, dict[int, torch.Tensor]]) -> dict[int, dict[int, torch.Tensor]]:
            return {
                timestep: {layer: tensor.detach().cpu() for layer, tensor in layers.items()}
                for timestep, layers in store.items()
            }

        return {
            "attn": _cpu(self.attn),
            "resnet": _cpu(self.resnet),
            "attn_layers": sorted(self.attn_layers),
            "resnet_layers": sorted(self.resnet_layers),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        r"""
        Restore features previously returned by `to_state`.

        Args:
            state (`dict`):
                Mapping produced by `to_state`.
        """

        if not isinstance(state, dict) or "attn" not in state:
            raise ValueError("`spatial_features` must be a dict created by MMDiffPipeline.")
        self.attn = {int(t): {int(i): v for i, v in layers.items()} for t, layers in state["attn"].items()}
        self.resnet = {
            int(t): {int(i): v for i, v in layers.items()} for t, layers in state.get("resnet", {}).items()
        }
        if "attn_layers" in state:
            self.attn_layers = set(int(i) for i in state["attn_layers"])
        if "resnet_layers" in state:
            self.resnet_layers = set(int(i) for i in state["resnet_layers"])


class MMDiffAttnProcessor(AttnProcessor):
    r"""
    Attention processor that records or replaces self-attention queries during spatial transfer.

    Args:
        layer_idx (`int`):
            1-based up-block self-attention index.
        store (`SpatialFeatureStore`):
            Shared feature store used by the current denoising loop.
    """

    def __init__(self, layer_idx: int, store: SpatialFeatureStore) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.store = store

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        injected = self.store.get_attn(self.layer_idx)
        if injected is not None:
            query = injected.to(device=query.device, dtype=query.dtype)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        self.store.save_attn(self.layer_idx, query)

        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


@contextmanager
def spatial_transfer_hooks(unet: UNet2DConditionModel, store: SpatialFeatureStore):
    r"""
    Install native attention processors and ResNet wrappers for one denoising run.

    Args:
        unet (`UNet2DConditionModel`):
            UNet to instrument.
        store (`SpatialFeatureStore`):
            Feature store read or written by the installed hooks.
    """

    attn_modules = collect_up_self_attentions(unet)
    resnet_modules = collect_up_resnets(unet)
    original_processors = [(module, module.processor) for module in attn_modules]
    original_forwards = []

    for layer_idx, module in enumerate(attn_modules, start=1):
        module.set_processor(MMDiffAttnProcessor(layer_idx, store))

    for layer_idx, module in enumerate(resnet_modules):
        original_forward = module.forward

        def _make_forward(orig: Callable, idx: int):
            def wrapped(hidden_states: torch.Tensor, temb: torch.Tensor | None = None, *args: Any, **kwargs: Any):
                output = orig(hidden_states, temb, *args, **kwargs)
                if store.mode == "save":
                    store.save_resnet(idx, output)
                elif store.mode == "inject":
                    injected = store.get_resnet(idx)
                    if injected is not None:
                        output = injected.to(device=output.device, dtype=output.dtype)
                return output

            return wrapped

        original_forwards.append((module, original_forward))
        module.forward = _make_forward(original_forward, layer_idx)

    try:
        yield store
    finally:
        for module, processor in original_processors:
            module.set_processor(processor)
        for module, original_forward in original_forwards:
            module.forward = original_forward


def normalize_modalities(modalities: str | list[str] | tuple[str, ...]) -> list[str]:
    r"""
    Validate and normalize the modality list passed to the pipeline.

    Args:
        modalities (`str` or sequence of `str`):
            `"all"` or any subset of `opt`, `sar`, and `ir`.

    Returns:
        `list[str]`: Deduplicated modalities in OPT → SAR → IR order.
    """

    if isinstance(modalities, str):
        requested = list(SUPPORTED_MODALITIES) if modalities.lower() == "all" else [modalities.lower()]
    else:
        requested = [str(item).lower() for item in modalities]

    unknown = [item for item in requested if item not in SUPPORTED_MODALITIES]
    if unknown:
        raise ValueError(
            f"Unsupported modalities {unknown}. Expected a subset of {list(SUPPORTED_MODALITIES)} or 'all'."
        )
    if not requested:
        raise ValueError("At least one modality must be requested.")

    ordered = [item for item in SUPPORTED_MODALITIES if item in requested]
    return ordered


@dataclass
class MMDiffPipelineOutput(BaseOutput):
    r"""
    Output of [`MMDiffPipeline`].

    Args:
        images (`list`):
            Images for the first requested modality, matching the Stable Diffusion `images` convention.
        opt (`list`, *optional*):
            Optical images when that modality was generated.
        sar (`list`, *optional*):
            SAR images when that modality was generated.
        ir (`list`, *optional*):
            Infrared images when that modality was generated.
        nsfw_content_detected (`list[bool]`, *optional*):
            Safety-checker flags for the primary `images` batch, if a checker is enabled.
        spatial_features (`dict`, *optional*):
            In-memory OPT features when `return_spatial_features=True`.
    """

    images: list[Any]
    opt: list[Any] | None = None
    sar: list[Any] | None = None
    ir: list[Any] | None = None
    nsfw_content_detected: list[bool] | None = None
    spatial_features: dict[str, Any] | None = None


class MMDiffPipeline(
    StableDiffusionPipeline,
    DiffusionPipeline,
    TextualInversionLoaderMixin,
    StableDiffusionLoraLoaderMixin,
    IPAdapterMixin,
    FromSingleFileMixin,
):
    r"""
    Pipeline for jointly generating spatially consistent optical, SAR, and infrared images.

    MMDiff fine-tunes a Stable Diffusion v1 UNet on optical remote-sensing pairs, then adapts
    SAR and IR style with LoRA. During inference the OPT branch records up-block self-attention
    queries and residual features; those features are injected into the SAR and IR branches.

    Parameters:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder used to decode latents into images.
        text_encoder ([`CLIPTextModel`]):
            Frozen CLIP text encoder.
        tokenizer ([`CLIPTokenizer`]):
            CLIP tokenizer paired with `text_encoder`.
        unet ([`UNet2DConditionModel`]):
            OPT-finetuned UNet. SAR/IR LoRA adapters are applied on top of this backbone.
        scheduler ([`KarrasDiffusionSchedulers`]):
            Denoising scheduler. The original sampling code uses [`DDPMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`], *optional*):
            Optional safety checker. Disabled by default for remote-sensing imagery.
        feature_extractor ([`CLIPImageProcessor`], *optional*):
            Feature extractor used only when `safety_checker` is enabled.
        image_encoder ([`CLIPVisionModelWithProjection`], *optional*):
            Optional IP-Adapter image encoder.
        requires_safety_checker (`bool`, *optional*, defaults to `False`):
            Whether a missing safety checker should emit a warning.
        lora_root (`str`, *optional*, defaults to `"loras"`):
            Directory (relative to the model root) that contains `sar/<scene>` and `ir/<scene>` adapters.
    """

    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
    _exclude_from_cpu_offload = ["safety_checker"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker | None = None,
        feature_extractor: CLIPImageProcessor | None = None,
        image_encoder: CLIPVisionModelWithProjection | None = None,
        requires_safety_checker: bool = False,
        lora_root: str = "loras",
    ) -> None:
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            requires_safety_checker=requires_safety_checker,
        )
        self.register_to_config(lora_root=lora_root, requires_safety_checker=requires_safety_checker)
        self.spatial_store = SpatialFeatureStore()
        self._loaded_scene: str | None = None

    def check_inputs(
        self,
        prompt: str | list[str] | None,
        height: int,
        width: int,
        callback_steps: int | None,
        negative_prompt: str | list[str] | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        ip_adapter_image: PipelineImageInput | None = None,
        ip_adapter_image_embeds: list[torch.Tensor] | None = None,
        callback_on_step_end_tensor_inputs: list[str] | None = None,
        modalities: str | list[str] | None = None,
        scene: str | None = None,
        spatial_features: dict[str, Any] | None = None,
    ) -> None:
        r"""
        Validate standard Stable Diffusion arguments plus MMDiff modality options.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                Text prompt. Required unless `prompt_embeds` is provided.
            height (`int`):
                Output height in pixels. Must be divisible by the VAE scale factor.
            width (`int`):
                Output width in pixels. Must be divisible by the VAE scale factor.
            callback_steps (`int`, *optional*):
                Deprecated callback interval forwarded to the parent checker.
            negative_prompt (`str` or `list[str]`, *optional*):
                Negative prompt used for classifier-free guidance.
            prompt_embeds (`torch.Tensor`, *optional*):
                Precomputed prompt embeddings.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Precomputed negative prompt embeddings.
            ip_adapter_image (`PipelineImageInput`, *optional*):
                Optional IP-Adapter image.
            ip_adapter_image_embeds (`list[torch.Tensor]`, *optional*):
                Optional precomputed IP-Adapter embeddings.
            callback_on_step_end_tensor_inputs (`list[str]`, *optional*):
                Tensor names forwarded to step-end callbacks.
            modalities (`str` or `list[str]`, *optional*):
                Requested modalities; validated by `normalize_modalities`.
            scene (`str`, *optional*):
                LoRA scene name used for SAR/IR adapters.
            spatial_features (`dict`, *optional*):
                Previously captured OPT features.
        """

        super().check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )
        if modalities is not None:
            normalize_modalities(modalities)
        if scene is not None and not isinstance(scene, str):
            raise TypeError(f"`scene` must be a string, got {type(scene)}.")
        if spatial_features is not None and not isinstance(spatial_features, dict):
            raise TypeError("`spatial_features` must be a dict produced by this pipeline.")

    def decode_latents(self, latents: torch.Tensor, single_channel: bool = False) -> torch.Tensor:
        r"""
        Decode latents with the VAE, optionally collapsing RGB to a single SAR/IR channel.

        Args:
            latents (`torch.Tensor`):
                Denoised latent tensor of shape `(batch, 4, h, w)`.
            single_channel (`bool`, *optional*, defaults to `False`):
                If `True`, average decoded RGB channels. This matches the original
                single-channel VAE decoder used for SAR and IR.

        Returns:
            `torch.Tensor`: Decoded images in `[-1, 1]`.
        """

        latents = latents / self.vae.config.scaling_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        if single_channel:
            image = image.mean(dim=1, keepdim=True)
        return image

    def resolve_lora_dir(self, modality: str, scene: str, lora_path: str | Path | None = None) -> Path:
        r"""
        Resolve the directory that stores a scene-specific SAR or IR LoRA.

        Args:
            modality (`str`):
                `"sar"` or `"ir"`.
            scene (`str`):
                Scene name such as `"ship"` or `"beach"`.
            lora_path (`str` or `Path`, *optional*):
                Explicit override. When omitted, `{model_root}/{lora_root}/{modality}/{scene}` is used.

        Returns:
            `Path`: Directory expected to contain `pytorch_lora_weights.safetensors`.
        """

        if lora_path is not None:
            return Path(lora_path)
        root = Path(self.config.lora_root)
        if not root.is_absolute():
            base = getattr(self, "name_or_path", None) or "."
            root = Path(base) / root
        return root / modality / scene

    def load_scene_loras(
        self,
        scene: str,
        sar_lora_path: str | Path | None = None,
        ir_lora_path: str | Path | None = None,
    ) -> None:
        r"""
        Load SAR and IR LoRA adapters for `scene` as named PEFT adapters.

        Args:
            scene (`str`):
                Scene used to resolve default LoRA directories.
            sar_lora_path (`str` or `Path`, *optional*):
                Explicit SAR adapter directory or weight file.
            ir_lora_path (`str` or `Path`, *optional*):
                Explicit IR adapter directory or weight file.
        """

        if self._loaded_scene == scene and sar_lora_path is None and ir_lora_path is None:
            return

        if hasattr(self, "unload_lora_weights"):
            try:
                self.unload_lora_weights()
            except Exception:
                logger.debug("No previously loaded LoRA adapters to unload.")

        loaded = False
        for modality, path in (("sar", sar_lora_path), ("ir", ir_lora_path)):
            adapter_dir = self.resolve_lora_dir(modality, scene, path)
            weight_file = adapter_dir if adapter_dir.is_file() else adapter_dir / "pytorch_lora_weights.safetensors"
            if not Path(weight_file).is_file() and not adapter_dir.is_dir():
                logger.warning("Skipping %s LoRA for scene '%s'; missing path: %s", modality, scene, adapter_dir)
                continue
            load_target = adapter_dir if adapter_dir.is_dir() else adapter_dir.parent
            self.load_lora_weights(str(load_target), adapter_name=modality)
            loaded = True

        if loaded:
            self._loaded_scene = scene

    def _set_modality_adapter(self, modality: str) -> None:
        if modality == "opt":
            if hasattr(self, "disable_lora"):
                try:
                    self.disable_lora()
                except Exception:
                    logger.debug("LoRA disable skipped; no adapters are active.")
            return
        if hasattr(self, "set_adapters"):
            try:
                self.set_adapters(modality)
            except Exception as error:
                logger.warning("Could not activate the '%s' LoRA adapter: %s", modality, error)

    def _postprocess_image(
        self,
        image: torch.Tensor,
        output_type: str,
        single_channel: bool,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Any:
        if output_type == "latent":
            return image
        if single_channel:
            image, has_nsfw = self.run_safety_checker(image.repeat(1, 3, 1, 1) if image.shape[1] == 1 else image, device, dtype)
            del has_nsfw
            image = image.mean(dim=1, keepdim=True)
            image = (image / 2 + 0.5).clamp(0, 1)
            image_np = image.cpu().permute(0, 2, 3, 1).float().numpy()
            if output_type == "np":
                return image_np[..., 0]
            if output_type == "pil":
                return [
                    self.numpy_to_pil(frame)[0].convert("L") if frame.ndim == 3 else self.numpy_to_pil(frame[..., None])[0]
                    for frame in image_np
                ]
            raise ValueError(f"Unknown output_type '{output_type}'. Use 'pil', 'np', or 'latent'.")

        image, has_nsfw_concept = self.run_safety_checker(image, device, dtype)
        do_denormalize = [True] * image.shape[0] if has_nsfw_concept is None else [not flag for flag in has_nsfw_concept]
        return self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

    def _denoise(
        self,
        prompt_embeds: torch.Tensor,
        timesteps: torch.Tensor,
        latents: torch.Tensor,
        extra_step_kwargs: dict[str, Any],
        timestep_cond: torch.Tensor | None,
        added_cond_kwargs: dict[str, Any] | None,
        store_mode: str,
        callback: Callable | None,
        callback_steps: int | None,
        callback_on_step_end: Callable | None,
        callback_on_step_end_tensor_inputs: list[str],
        num_inference_steps: int,
    ) -> torch.Tensor:
        store = self.spatial_store
        store.mode = store_mode
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with spatial_transfer_hooks(self.unet, store), self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                store.set_timestep(t)
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                if hasattr(self.scheduler, "scale_model_input"):
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    if self.guidance_rescale > 0.0:
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {name: locals()[name] for name in callback_on_step_end_tensor_inputs if name in locals()}
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and callback_steps is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        store.mode = "off"
        return latents

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: str | list[str] | None = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        timesteps: list[int] | None = None,
        sigmas: list[float] | None = None,
        guidance_scale: float = 7.5,
        negative_prompt: str | list[str] | None = None,
        num_images_per_prompt: int | None = 1,
        eta: float = 0.0,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        ip_adapter_image: PipelineImageInput | None = None,
        ip_adapter_image_embeds: list[torch.Tensor] | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: dict[str, Any] | None = None,
        guidance_rescale: float = 0.0,
        clip_skip: int | None = None,
        callback_on_step_end: Callable[..., Any] | None = None,
        callback_on_step_end_tensor_inputs: list[str] | None = None,
        modalities: str | list[str] = "all",
        scene: str = DEFAULT_SCENE,
        sar_lora_path: str | Path | None = None,
        ir_lora_path: str | Path | None = None,
        attn_layers: list[int] | tuple[int, ...] | None = None,
        resnet_layers: list[int] | tuple[int, ...] | None = None,
        resnet_time: float = 1.0,
        spatial_features: dict[str, Any] | None = None,
        return_spatial_features: bool = False,
        **kwargs: Any,
    ) -> MMDiffPipelineOutput | tuple:
        r"""
        Generate optical, SAR, and/or infrared images from one text prompt.

        The call follows the Stable Diffusion stage order: check inputs, define call
        parameters, encode the prompt, prepare timesteps, prepare latents, prepare extra
        step kwargs, then run the denoising loop. OPT is generated first so its spatial
        features can be transferred into the SAR and IR branches.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                Text prompt that guides all requested modalities.
            height (`int`, *optional*, defaults to `256`):
                Output height in pixels. MMDiff was trained at 256×256.
            width (`int`, *optional*, defaults to `256`):
                Output width in pixels.
            num_inference_steps (`int`, *optional*, defaults to `50`):
                Number of denoising steps.
            timesteps (`list[int]`, *optional*):
                Custom descending timestep schedule.
            sigmas (`list[float]`, *optional*):
                Custom sigma schedule for compatible schedulers.
            guidance_scale (`float`, *optional*, defaults to `7.5`):
                Classifier-free guidance scale. Guidance is enabled when this value is `> 1`.
            negative_prompt (`str` or `list[str]`, *optional*):
                Prompt used for the unconditional branch. Defaults to empty strings.
            num_images_per_prompt (`int`, *optional*, defaults to `1`):
                Number of images drawn per prompt.
            eta (`float`, *optional*, defaults to `0.0`):
                DDIM eta. Ignored by schedulers that do not accept `eta`.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                RNG used to sample the shared initial latents for every modality.
            latents (`torch.Tensor`, *optional*):
                Optional pre-sampled latents reused for every modality.
            prompt_embeds (`torch.Tensor`, *optional*):
                Precomputed prompt embeddings.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Precomputed unconditional embeddings.
            ip_adapter_image (`PipelineImageInput`, *optional*):
                Optional IP-Adapter image condition.
            ip_adapter_image_embeds (`list[torch.Tensor]`, *optional*):
                Optional precomputed IP-Adapter embeddings.
            output_type (`str`, *optional*, defaults to `"pil"`):
                `"pil"`, `"np"`, or `"latent"`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return [`MMDiffPipelineOutput`].
            cross_attention_kwargs (`dict`, *optional*):
                Extra kwargs forwarded to attention processors.
            guidance_rescale (`float`, *optional*, defaults to `0.0`):
                Optional guidance rescale factor.
            clip_skip (`int`, *optional*):
                Number of CLIP layers to skip when encoding prompts.
            callback_on_step_end (`Callable`, *optional*):
                Optional per-step callback.
            callback_on_step_end_tensor_inputs (`list[str]`, *optional*):
                Tensor names passed to `callback_on_step_end`.
            modalities (`str` or `list[str]`, *optional*, defaults to `"all"`):
                `"all"` or any subset of `"opt"`, `"sar"`, `"ir"`.
            scene (`str`, *optional*, defaults to `"ship"`):
                Scene used to resolve packaged SAR/IR LoRA adapters.
            sar_lora_path (`str` or `Path`, *optional*):
                Override for the SAR LoRA directory or weight file.
            ir_lora_path (`str` or `Path`, *optional*):
                Override for the IR LoRA directory or weight file.
            attn_layers (`list[int]`, *optional*):
                1-based up-block self-attention layers to transfer. Defaults to `1..9`.
            resnet_layers (`list[int]`, *optional*):
                0-based up-block ResNet layers to transfer. Defaults to `(2,)`.
            resnet_time (`float`, *optional*, defaults to `1.0`):
                Fraction of the early timestep schedule that receives ResNet injection.
            spatial_features (`dict`, *optional*):
                Features from a previous OPT run. When omitted, OPT is run first whenever
                SAR or IR generation needs transfer features.
            return_spatial_features (`bool`, *optional*, defaults to `False`):
                If `True`, include the captured OPT features in the output.

        Examples:

        Returns:
            [`MMDiffPipelineOutput`] or `tuple`:
                Generated images grouped by modality. `images` is the first requested modality.
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)
        callback_on_step_end_tensor_inputs = callback_on_step_end_tensor_inputs or ["latents"]
        requested = normalize_modalities(modalities)
        attn_layers = tuple(DEFAULT_ATTN_LAYERS if attn_layers is None else attn_layers)
        resnet_layers = tuple(DEFAULT_RESNET_LAYERS if resnet_layers is None else resnet_layers)

        height = DEFAULT_RESOLUTION if height is None else height
        width = DEFAULT_RESOLUTION if width is None else width

        # 1. Check inputs
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
            modalities=requested,
            scene=scene,
            spatial_features=spatial_features,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        num_images_per_prompt = 1 if num_images_per_prompt is None else num_images_per_prompt
        device = self._execution_device

        needs_transfer = any(modality in requested for modality in ("sar", "ir"))
        run_opt = "opt" in requested or (needs_transfer and spatial_features is None)
        if needs_transfer and not run_opt and spatial_features is None:
            raise ValueError("SAR/IR generation requires OPT spatial features. Run OPT first or pass `spatial_features`.")

        if any(modality in requested for modality in ("sar", "ir")):
            self.load_scene_loras(scene, sar_lora_path=sar_lora_path, ir_lora_path=ir_lora_path)

        self.spatial_store.reset()
        self.spatial_store.attn_layers = set(attn_layers)
        self.spatial_store.resnet_layers = set(resnet_layers)
        if spatial_features is not None:
            self.spatial_store.load_state(spatial_features)

        # 3. Encode input condition
        lora_scale = self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        added_cond_kwargs = None
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )
            added_cond_kwargs = {"image_embeds": image_embeds}

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )
        timestep_values = [int(step) for step in timesteps]
        self.spatial_store.inject_attn_timesteps = set(timestep_values)
        cutoff = max(1, int(len(timestep_values) * resnet_time)) if resnet_time > 0 else 0
        self.spatial_store.inject_resnet_timesteps = set(timestep_values[:cutoff])

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        init_latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        timestep_cond = None
        if getattr(self.unet.config, "time_cond_proj_dim", None) is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=init_latents.dtype)

        # 7. Run denoising loop for each requested branch
        generated: dict[str, Any] = {}
        primary_nsfw = None
        for modality in (("opt",) if run_opt else ()) + tuple(item for item in requested if item != "opt"):
            self._set_modality_adapter(modality)
            store_mode = "save" if modality == "opt" else "inject"
            latents_in = init_latents.clone()
            latents_out = self._denoise(
                prompt_embeds=prompt_embeds,
                timesteps=timesteps,
                latents=latents_in,
                extra_step_kwargs=extra_step_kwargs,
                timestep_cond=timestep_cond,
                added_cond_kwargs=added_cond_kwargs,
                store_mode=store_mode,
                callback=callback,
                callback_steps=callback_steps,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                num_inference_steps=num_inference_steps,
            )
            if modality not in requested:
                continue
            if output_type == "latent":
                images = latents_out
                has_nsfw = None
            else:
                single_channel = modality in {"sar", "ir"}
                decoded = self.decode_latents(latents_out, single_channel=single_channel)
                if single_channel:
                    images = self._postprocess_image(
                        decoded, output_type=output_type, single_channel=True, dtype=prompt_embeds.dtype, device=device
                    )
                    has_nsfw = None
                else:
                    images, has_nsfw = self.run_safety_checker(decoded, device, prompt_embeds.dtype)
                    do_denormalize = [True] * images.shape[0] if has_nsfw is None else [not flag for flag in has_nsfw]
                    images = self.image_processor.postprocess(
                        images, output_type=output_type, do_denormalize=do_denormalize
                    )
            generated[modality] = images
            if primary_nsfw is None:
                primary_nsfw = has_nsfw

        self.maybe_free_model_hooks()
        feature_state = self.spatial_store.to_state() if return_spatial_features else None
        images = generated.get(requested[0])
        if not return_dict:
            return (images, primary_nsfw)

        return MMDiffPipelineOutput(
            images=images,
            opt=generated.get("opt"),
            sar=generated.get("sar"),
            ir=generated.get("ir"),
            nsfw_content_detected=primary_nsfw,
            spatial_features=feature_state,
        )


__all__ = ["MMDiffPipeline", "MMDiffPipelineOutput", "SpatialFeatureStore"]
