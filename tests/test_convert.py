from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.convert_mmdiff_to_diffusers import (
    convert,
    detect_scenes,
    find_lora_dir,
)
from tests.tiny_helpers import make_dummy_source_tree


def _args(**overrides) -> Namespace:
    payload = dict(
        source="",
        dump_path="",
        opt_subdir="mmdiff-opt",
        sar_subdir="mmdiff-sar",
        ir_subdir="mmdiff-ir",
        opt_checkpoint=None,
        from_ckpt=None,
        scenes=None,
        lora_root="loras",
        scheduler_class="DDPMScheduler",
        skip_pipeline_copy=False,
        overwrite=False,
    )
    payload.update(overrides)
    return Namespace(**payload)


def test_detect_and_find_scenes(tmp_path: Path) -> None:
    source = make_dummy_source_tree(tmp_path / "src")
    scenes = detect_scenes(source / "mmdiff-sar", "sd-sar-lora-")
    assert scenes == ["beach", "ship"]
    found = find_lora_dir(source / "mmdiff-sar", "sd-sar-lora-", "ship")
    assert found is not None
    assert found.name == "sd-sar-lora-ship"


def test_convert_packages_pipeline_layout(tmp_path: Path) -> None:
    source = make_dummy_source_tree(tmp_path / "src")
    dump = tmp_path / "mmdiff-diffusers"
    args = _args(source=str(source), dump_path=str(dump))
    convert(args)

    assert (dump / "pipeline.py").exists()
    index = json.loads((dump / "model_index.json").read_text(encoding="utf-8"))
    assert index["_class_name"] == "MMDiffPipeline"
    assert index["scheduler"] == ["diffusers", "DDPMScheduler"]
    assert index["safety_checker"] == [None, None]
    assert index["lora_root"] == "loras"

    scheduler = json.loads((dump / "scheduler" / "scheduler_config.json").read_text(encoding="utf-8"))
    assert scheduler["_class_name"] == "DDPMScheduler"
    assert "trained_betas" not in scheduler or scheduler["trained_betas"] is None

    assert (dump / "loras" / "sar" / "ship" / "pytorch_lora_weights.safetensors").exists()
    assert (dump / "loras" / "ir" / "beach" / "pytorch_lora_weights.safetensors").exists()
    assert not (dump / "loras" / "sar" / "ship" / "checkpoint-15000").exists()
    assert not (dump / "unet" / "from_ckpt").exists()


def test_convert_uses_opt_training_checkpoint(tmp_path: Path) -> None:
    source = make_dummy_source_tree(tmp_path / "src")
    dump = tmp_path / "out"
    args = _args(
        source=str(source),
        dump_path=str(dump),
        opt_checkpoint="checkpoint-15000",
        scenes="ship",
        overwrite=True,
        skip_pipeline_copy=True,
    )
    convert(args)
    unet_config = json.loads((dump / "unet" / "config.json").read_text(encoding="utf-8"))
    assert unet_config == {"from": "ckpt"}
    assert not (dump / "pipeline.py").exists()
    assert (dump / "loras" / "sar" / "ship").exists()
    assert not (dump / "loras" / "sar" / "beach").exists()


def test_convert_refuses_to_overwrite(tmp_path: Path) -> None:
    source = make_dummy_source_tree(tmp_path / "src")
    dump = tmp_path / "out"
    dump.mkdir()
    (dump / "keep.txt").write_text("x", encoding="utf-8")
    args = _args(source=str(source), dump_path=str(dump), scenes="ship", skip_pipeline_copy=True)
    with pytest.raises(FileExistsError):
        convert(args)
