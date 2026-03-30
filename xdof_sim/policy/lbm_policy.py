"""LBM policy for sim inference.

Requires the model architecture package to be installed or on PYTHONPATH.
This module imports models.lbm.DiTPolicy.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from termcolor import cprint

from xdof_sim.policy.base import PolicyConfig, BasePolicy


def _resize_with_pad_torch(images, height, width, mode="bilinear"):
    """Resize images with padding to preserve aspect ratio."""
    import torch.nn.functional as F

    if images.shape[-1] <= 4:
        channels_last = True
        images = images.permute(0, 3, 1, 2)
    else:
        channels_last = False

    batch_size, channels, cur_height, cur_width = images.shape
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    antialias = mode == "bilinear"
    resized_images = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
        antialias=antialias,
    )

    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(0, 1)

    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),
        mode="constant",
        value=0,
    )

    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)
    return padded_images


def _normalize_image(im):
    """ImageNet normalization."""
    _MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=im.device)
    _STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=im.device)
    _MEAN = _MEAN.view(1, 3, 1, 1)
    _STD = _STD.view(1, 3, 1, 1)
    return (im - _MEAN) / (_STD + 1e-6)


@dataclass
class LBMPolicyConfig(PolicyConfig):
    """Config for LBM-family policies."""

    camera_keys: Optional[List[str]] = None
    tau: float = 1.0
    num_diffusion_steps: int = 10
    use_torch_compile: bool = False
    deterministic: bool = False
    diffusion_steps: int = 10
    sampler: str = "euler"
    num_bins: int = 51
    v_min: float = -3.0
    v_max: float = 3.0
    prediction_type: str = "velocity"


class LBMPolicy(BasePolicy):
    """Policy for LBM-family models."""

    cfg: LBMPolicyConfig

    def __init__(self, cfg: LBMPolicyConfig):
        self.tau = cfg.tau
        self.deterministic = cfg.deterministic
        if self.deterministic:
            import os as _os

            _os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        self.sampler = cfg.sampler
        super().__init__(cfg)

    def _build_model(self, ckpt_path: str) -> None:
        model_cfgs = {
            "dit_S": {"hidden_size": 384, "depth": 12, "num_heads": 6},
            "dit_B": {"hidden_size": 768, "depth": 12, "num_heads": 12},
            "dit_L": {"hidden_size": 1024, "depth": 24, "num_heads": 16},
            "dit_xL": {"hidden_size": 1536, "depth": 32, "num_heads": 24},
        }
        mcfg = model_cfgs.get(self.cfg.model_size, model_cfgs["dit_B"])
        camera_keys = (
            self.cfg.camera_keys
            if self.cfg.camera_keys is not None
            else ["top", "left", "right"]
        )

        if self.cfg.policy_type == "lbm":
            from models.lbm import DiTPolicy  # type: ignore[import]

            self.model = DiTPolicy(
                state_dim=14,
                action_dim=14,
                chunk_length=self.chunk_len,
                img_emb_dim=512,
                hidden_size=mcfg["hidden_size"],
                depth=mcfg["depth"],
                num_heads=mcfg["num_heads"],
                task_encoder="clip",
                camera_keys=camera_keys,
                num_diffusion_timesteps=self.cfg.diffusion_steps,
                device=self.device,
                post_cond_layer_norm=True,
            ).to(self.device)

        elif self.cfg.policy_type == "lbm_ce":
            from models.lbm_distributional import DiTPolicyCE  # type: ignore[import]

            self.model = DiTPolicyCE(
                state_dim=14,
                action_dim=14,
                chunk_length=self.chunk_len,
                img_emb_dim=512,
                hidden_size=mcfg["hidden_size"],
                depth=mcfg["depth"],
                num_heads=mcfg["num_heads"],
                task_encoder="clip",
                camera_keys=camera_keys,
                num_diffusion_timesteps=self.cfg.diffusion_steps,
                device=self.device,
                post_cond_layer_norm=True,
                num_bins=self.cfg.num_bins,
                v_min=self.cfg.v_min,
                v_max=self.cfg.v_max,
                prediction_type=self.cfg.prediction_type,
            ).to(self.device)

        else:
            raise ValueError(f"Unsupported LBM policy type: {self.cfg.policy_type}")

    def _post_load_setup(self) -> None:
        if bool(getattr(self.cfg, "use_torch_compile", False)) and hasattr(
            torch, "compile"
        ):
            try:
                torch.set_float32_matmul_precision("high")
                self.model_infer = torch.compile(
                    self.model.infer, mode="max-autotune", fullgraph=True
                )
                self.model.to(torch.bfloat16)
                try:
                    self.model.clip_model.to(torch.float32)
                    if hasattr(self.model, "enable_clip_compile"):
                        self.model.enable_clip_compile(mode="default", fullgraph=False)
                except Exception:
                    pass
            except Exception as e:
                cprint(f"torch.compile failed; falling back to eager: {e}", "red")

    def _infer_actions(
        self, inputs, obs, images, *, noise=None, action_prefix=None, prefix_length=None
    ):
        def _to_tensor(v):
            if not isinstance(v, torch.Tensor):
                v = torch.from_numpy(v.copy())
            v = v.float().to(self.device)
            if v.max() > 1.0:
                v = v / 255.0
            return v

        inputs["images"] = {
            k: _normalize_image(
                _resize_with_pad_torch(_to_tensor(v).unsqueeze(0), 224, 224)
            )
            for k, v in images.items()
        }

        action_prefix_tensor = None
        prefix_len = None
        if action_prefix is not None:
            action_prefix_norm = self.normalizer({"actions": action_prefix})["actions"]
            prefix_len = (
                len(action_prefix_norm) if prefix_length is None else prefix_length
            )
            action_prefix_padded = np.zeros(
                (self.chunk_len, self.action_dim), dtype=np.float32
            )
            action_prefix_padded[:prefix_len] = action_prefix_norm[:prefix_len]
            action_prefix_tensor = (
                torch.from_numpy(action_prefix_padded)
                .float()
                .to(self.device)
                .unsqueeze(0)
            )

        infer_params = inspect.signature(self.model.infer).parameters
        infer_kwargs = {"num_steps": self.model.num_diffusion_timesteps}
        if "tau" in infer_params:
            infer_kwargs["tau"] = self.tau
        if "sampler" in infer_params:
            infer_kwargs["sampler"] = self.sampler
        if "action_prefix" in infer_params and action_prefix_tensor is not None:
            infer_kwargs["action_prefix"] = action_prefix_tensor
            infer_kwargs["prefix_length"] = prefix_len

        if "noise" in infer_params:
            if noise is None:
                noise = torch.randn(
                    1, self.chunk_len, self.action_dim, device=self.device
                )
            else:
                noise = (
                    torch.from_numpy(noise).float().to(self.device).unsqueeze(0)
                )
            infer_kwargs["noise"] = noise
        else:
            noise = None

        actions = self.model.infer(inputs, **infer_kwargs)
        extras = {}
        if noise is not None:
            extras["noise"] = noise.cpu().numpy()[0]
        return actions.detach().cpu().numpy()[0], extras
