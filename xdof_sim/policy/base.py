"""Base policy interface for running trained models in sim."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from termcolor import cprint

from xdof_sim.transforms import Normalize, Unnormalize


def _download_wandb_checkpoint(
    artifact_ref: str, root: str = "checkpoints"
) -> Optional[str]:
    """Download a W&B artifact checkpoint. Requires wandb to be installed."""
    import wandb

    artifact_name = artifact_ref.split(":")[0].split("/")[-1]
    version = artifact_ref.split(":")[1]

    if version != "latest":
        download_path = os.path.join(root, artifact_name, f"version_{version}")
        if os.path.exists(download_path):
            return download_path

    api = wandb.Api()
    art = api.artifact(artifact_ref, type="checkpoint")
    version = art.version
    download_path = os.path.join(root, artifact_name, f"version_{version}")
    if os.path.exists(download_path):
        return download_path

    os.makedirs(download_path, exist_ok=True)
    print(f"Downloading artifact to {download_path}")
    art.download(root=download_path)
    return download_path


@dataclass
class PolicyConfig:
    """Shared config fields used by all policy types."""

    config: Optional[str] = None
    dir: Optional[str] = None
    ckpt_path: Optional[str] = None
    policy_type: Optional[str] = "lbm"
    model_size: Optional[str] = "dit_B"
    state_dim: Optional[int] = 14
    action_dim: Optional[int] = 14
    chunk_len: Optional[int] = 30
    mask_state: Optional[bool] = False
    unnormalize_actions: Optional[bool] = True
    norm_stats_path: Optional[str] = None


class BasePolicy:
    def __init__(self, cfg: PolicyConfig):
        self.cfg = cfg
        if self.cfg.ckpt_path is None and self.cfg.dir is not None:
            self.cfg.ckpt_path = self.cfg.dir
        self.device = "cuda:0"
        self._task_vec_h = None
        self._task_str: Optional[str] = None
        self.chunk_len = cfg.chunk_len
        self.action_dim = 14
        self.use_quantile_norm = False

        ckpt_path = self._resolve_checkpoint()
        print(f"Loading {str(self.cfg.policy_type).upper()} {self.cfg.model_size}...")
        self._build_model(ckpt_path)
        cprint(
            f"Loaded {str(self.cfg.policy_type).upper()} {self.cfg.model_size} "
            f"({self.model.compute_num_parameters()/1e6:.2f}M params)",
            "cyan",
        )
        ckpt = self._load_weights(ckpt_path)
        self._post_load_setup()
        self._load_norm_stats(ckpt)
        self.normalizer = Normalize(
            norm_stats=self.norm_stats,
            use_quantiles=self.use_quantile_norm,
            strict=False,
        )
        self.unnormalizer = Unnormalize(
            norm_stats=self.norm_stats,
            use_quantiles=self.use_quantile_norm,
            strict=False,
        )

    def _resolve_checkpoint(self) -> str:
        if self.cfg.ckpt_path is None:
            raise ValueError("No checkpoint provided")
        checkpoints_root = "checkpoints"
        if self.cfg.ckpt_path.startswith("far-wandb/FAR-abc"):
            artifact_path = _download_wandb_checkpoint(
                artifact_ref=self.cfg.ckpt_path, root=checkpoints_root
            )
            ckpt_path = os.path.join(artifact_path, "last.pt")
            if not os.path.exists(ckpt_path):
                pts = [p for p in os.listdir(artifact_path) if p.endswith(".pt")]
                if not pts:
                    raise FileNotFoundError(
                        f"Could not find any .pt in {artifact_path}"
                    )
                ckpt_path = os.path.join(artifact_path, pts[0])
        else:
            ckpt_path = self.cfg.ckpt_path
        return ckpt_path

    def _load_weights(self, ckpt_path: str) -> dict:
        ckpt = torch.load(ckpt_path, map_location="cpu")

        if "model" in ckpt:
            state_dict = ckpt["model"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt["state_dict"]

        state_dict = {
            k.removeprefix("_orig_mod."): v for k, v in state_dict.items()
        }
        self.model.load_state_dict(state_dict)
        self.max_episode_length = ckpt.get("max_episode_length")
        self.model.eval()
        return ckpt

    def _post_load_setup(self) -> None:
        pass

    def _load_norm_stats(self, ckpt: dict) -> None:
        if self.cfg.norm_stats_path is not None:
            with open(self.cfg.norm_stats_path, "r") as f:
                norm_stats_data = json.load(f)
            self.norm_stats = norm_stats_data.get("norm_stats", norm_stats_data)
        elif "norm_stats" in ckpt and ckpt["norm_stats"] is not None:
            self.norm_stats = ckpt["norm_stats"]
            if "xdof" in self.norm_stats:
                self.norm_stats = self.norm_stats["xdof"]
        else:
            raise ValueError(
                "No norm stats found. Provide --norm-stats-path or ensure "
                "checkpoint contains norm_stats."
            )

    def set_task(self, task: Optional[str]) -> None:
        task_str = None if task is None else str(task).strip()
        if task_str is None or task_str == "":
            self._task_vec_h = None
            self._task_str = None
            return
        try:
            self._task_vec_h = self.model.encode_task_to_hidden(
                [task_str], device=self.device
            )
            self._task_str = task_str
        except Exception as e:
            print(f"Failed to compute task embedding for '{task}': {e}")
            self._task_vec_h = None
            self._task_str = None

    @torch.no_grad()
    def infer(
        self,
        obs: dict,
        *,
        noise: np.ndarray | None = None,
        action_prefix: np.ndarray | None = None,
        prefix_length: int | None = None,
    ) -> dict:
        """Run inference on the policy."""
        task_raw = str(obs["prompt"])
        task_trimmed = task_raw.strip()
        desired_task = None if task_trimmed == "" else task_trimmed
        if self._task_str != desired_task:
            self.set_task(desired_task)

        state_np = obs["state"]
        inputs = {
            "state": state_np,
            "actions": np.zeros(
                (self.chunk_len, self.action_dim), dtype=np.float32
            ),
        }
        inputs = self.normalizer(inputs)
        inputs = {
            "state": torch.from_numpy(inputs["state"])
            .float()
            .to(self.device)
            .unsqueeze(0),
            "prompt": [str(obs["prompt"])],
        }

        images = obs.get("images", {})
        if isinstance(images, dict):
            images = dict(images)
        else:
            images = {}

        result = self._infer_actions(
            inputs,
            obs,
            images,
            noise=noise,
            action_prefix=action_prefix,
            prefix_length=prefix_length,
        )

        if isinstance(result, tuple):
            actions, infer_extras = result
        else:
            actions, infer_extras = result, {}

        actions = self.unnormalizer({"actions": actions})["actions"]
        return {"actions": actions, **infer_extras}

    def _build_model(self, ckpt_path: str) -> None:
        raise NotImplementedError

    def _infer_actions(
        self, inputs, obs, images, *, noise=None, action_prefix=None, prefix_length=None
    ):
        raise NotImplementedError
