import torch
import numpy as np
from typing import Optional
import time
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model', 'diffusion-vs-ar', 'src'))

from transformers import GPT2LMHeadModel, GPT2Config


class DiffusionModelWrapper:
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        config_path = os.path.join(self.checkpoint_path, "config.json")
        model_path = os.path.join(self.checkpoint_path, "pytorch_model.bin")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        config = GPT2Config.from_json_file(config_path)

        base_model = GPT2LMHeadModel(config)

        state_dict = torch.load(model_path, map_location=self.device)

        if any(k.startswith('model.') for k in state_dict.keys()):
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    new_key = k[6:]
                    new_state_dict[new_key] = v
                elif k.startswith('denoise_model.'):
                    new_key = k.replace('denoise_model.', 'transformer.')
                    new_state_dict[new_key] = v
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict

        try:
            base_model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Warning: Could not load full state dict: {e}")
            base_model.load_state_dict(state_dict, strict=False)

        self.model = base_model
        self.model.to(self.device)
        self.model.eval()

        self.vocab = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                      "*", "+", "-", "/", "=", ",", "a", "b", "c", "d",
                      "e", "f", "g", "h", "i", "j"]

    def get_logits(self, grid: np.ndarray, timestep: int = 10) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not loaded")

        start_time = time.time()

        if isinstance(grid, str):
            grid = np.array([int(c) if c.isdigit() else 0 for c in grid], dtype=np.int64)

        grid = np.asarray(grid, dtype=np.int64)
        if grid.shape == (9, 9):
            grid = grid.flatten()
        assert grid.shape == (81,), f"Grid shape must be (81,), got {grid.shape}"

        grid_batch = np.expand_dims(grid, axis=0)
        input_ids = torch.from_numpy(grid_batch).to(device=self.device, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None
            )
            logits = outputs.logits

        digit_logits = logits[0, :, 1:10]

        digit_logits_np = digit_logits.cpu().numpy()

        self._last_inference_time = (time.time() - start_time) * 1000

        return digit_logits_np

    def _forward_digit_logits(self, flat_grid: np.ndarray) -> np.ndarray:
        grid_batch = np.expand_dims(flat_grid, axis=0)
        input_ids = torch.from_numpy(grid_batch).to(device=self.device, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None
            )
            logits = outputs.logits
        return logits[0, :, 1:10].cpu().numpy()

    def _topk_masking(self, scores: np.ndarray, cutoff_len: int) -> np.ndarray:
        if cutoff_len <= 0:
            return np.zeros_like(scores, dtype=bool)
        order = np.argsort(scores)
        mask = np.zeros_like(scores, dtype=bool)
        cutoff_len = min(cutoff_len, len(scores))
        mask[order[:cutoff_len]] = True
        return mask

    def get_logits_mdm_iterative(
        self,
        grid: np.ndarray,
        diffusion_steps: int = 8,
        decoding_strategy: str = "deterministic-cosine",
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not loaded")

        start_time = time.time()
        if isinstance(grid, str):
            grid = np.array([int(c) if c.isdigit() else 0 for c in grid], dtype=np.int64)
        working = np.asarray(grid, dtype=np.int64).copy()
        if working.shape == (9, 9):
            working = working.flatten()
        if working.shape != (81,):
            raise ValueError(f"Grid shape must be (81,), got {working.shape}")

        diffusion_steps = max(1, int(diffusion_steps))
        src_mask = working != 0
        init_maskable = ~src_mask
        xt = working.copy()
        xt[init_maskable] = self.mask_token_id
        final_logits = None

        try:
            _, schedule = decoding_strategy.split("-", 1)
        except ValueError:
            schedule = "cosine"
        if schedule not in {"linear", "cosine"}:
            schedule = "cosine"

        for t in range(diffusion_steps - 1, -1, -1):
            logits = self._forward_digit_logits(xt)
            final_logits = logits

            probs = torch.softmax(torch.from_numpy(logits.astype(np.float32)), dim=-1).numpy()
            x0 = np.argmax(probs, axis=-1) + 1
            x0_full = xt.copy()
            x0_full[init_maskable] = x0[init_maskable]

            if t == 0:
                xt = x0_full
                continue

            if schedule == "linear":
                rate = float(t) / float(diffusion_steps)
            else:
                rate = float(np.cos((diffusion_steps - t) / diffusion_steps * np.pi * 0.5))

            cutoff_len = int(np.floor(int(np.sum(init_maskable)) * rate))
            if cutoff_len <= 0:
                xt = x0_full
                continue

            conf = np.full((81,), 1000.0, dtype=np.float64)
            idx = np.where(init_maskable)[0]
            pred_idx = x0[idx] - 1
            conf[idx] = probs[idx, pred_idx]
            remask = self._topk_masking(conf, cutoff_len)

            xt = x0_full.copy()
            xt[remask] = self.mask_token_id

        self._last_inference_time = (time.time() - start_time) * 1000
        if final_logits is None:
            final_logits = self._forward_digit_logits(working)
        return final_logits

    def get_logits_batch(self, grids: np.ndarray, timestep: int = 10) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not loaded")

        start_time = time.time()

        if isinstance(grids, list):
            grids = np.array(grids, dtype=np.int64)

        grids = np.asarray(grids, dtype=np.int64)

        if grids.ndim == 2 and grids.shape[1] == 81:
            pass
        elif grids.ndim == 3 and grids.shape[1:] == (9, 9):
            grids = grids.reshape(grids.shape[0], 81)
        else:
            raise ValueError(f"Invalid grids shape: {grids.shape}")

        input_ids = torch.from_numpy(grids).to(device=self.device, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None
            )
            logits = outputs.logits

        digit_logits = logits[:, :, 1:10]

        digit_logits_np = digit_logits.cpu().numpy()

        self._last_inference_time = (time.time() - start_time) * 1000

        return digit_logits_np

    def get_last_inference_time_ms(self) -> float:
        return getattr(self, "_last_inference_time", 0.0)

    @property
    def mask_token_id(self) -> int:
        return 0

    def grid_to_string(self, grid: np.ndarray) -> str:
        return "".join(str(int(v)) for v in grid)

    def string_to_grid(self, s: str) -> np.ndarray:
        return np.array([int(c) if c.isdigit() else 0 for c in s[:81]], dtype=np.int64)
