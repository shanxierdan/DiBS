#!/usr/bin/env python3
"""Train neural family guidance models for Table4 extension."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA_ROOT = REPO_ROOT / "dataset" / "table4_extension"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "model" / "diffusion-vs-ar" / "output" / "extension"

SUDOKU_MAX_LEN = 25 * 25
SUDOKU_MAX_SYMBOL = 25
SUDOKU_MASK_TOKEN = SUDOKU_MAX_SYMBOL + 1
NQUEENS_MAX_N = 32
NQUEENS_MASK_TOKEN = NQUEENS_MAX_N


def parse_sizes_arg(sizes_arg: str, family: str) -> List[str]:
    if sizes_arg.strip():
        return [s.strip() for s in sizes_arg.split(",") if s.strip()]
    return ["4x4", "16x16", "25x25"] if family == "gen_sudoku" else ["8", "9", "10"]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def char_to_symbol_value(ch: str, n: int) -> int:
    if ch == "0" or ch == ".":
        return 0
    if n <= 16:
        alphabet = "0123456789ABCDEFG"
        idx = alphabet.find(ch.upper())
        if idx >= 0:
            return idx
    if ch.isdigit():
        return int(ch)
    return ord(ch.upper()) - ord("A") + 1


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def difficulty_weight(
    difficulty: str,
    easy_weight: float,
    medium_weight: float,
    hard_weight: float,
) -> float:
    d = (difficulty or "").strip().lower()
    if d in ("very_hard", "hard"):
        return hard_weight
    if d == "medium":
        return medium_weight
    return easy_weight


class SudokuTrainDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        sizes: List[str],
        split: str,
        easy_weight: float,
        medium_weight: float,
        hard_weight: float,
        unknown_ratio_weight: float,
    ):
        self.samples = []
        self.sample_weights: List[float] = []
        for size in sizes:
            split_path = data_root / "generalized_sudoku" / size / f"{split}.jsonl"
            if not split_path.exists():
                print(f"[warn] missing split file, skip: {split_path}")
                continue
            rows = load_jsonl(split_path)
            n = int(size.split("x")[0])
            seq_len = n * n
            for row in rows:
                puzzle = row["puzzle"][:seq_len]
                solution = row["solution"][:seq_len]
                x0 = [char_to_symbol_value(c, n) for c in solution]
                src = [1 if puzzle[i] not in ("0", ".") else 0 for i in range(seq_len)]
                unknown_cnt = sum(1 for v in src if v == 0)
                diff = row.get("difficulty", "easy")
                self.samples.append((x0, src, seq_len, n, diff))
                base_w = difficulty_weight(diff, easy_weight, medium_weight, hard_weight)
                unknown_ratio = unknown_cnt / float(max(1, seq_len))
                self.sample_weights.append(base_w * (1.0 + unknown_ratio_weight * unknown_ratio))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int], int, int, str]:
        return self.samples[idx]


class NQueensTrainDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        sizes: List[str],
        split: str,
        easy_weight: float,
        medium_weight: float,
        hard_weight: float,
        unknown_ratio_weight: float,
    ):
        self.samples = []
        self.sample_weights: List[float] = []
        for size in sizes:
            split_path = data_root / "nqueens" / size / f"{split}.jsonl"
            if not split_path.exists():
                print(f"[warn] missing split file, skip: {split_path}")
                continue
            rows = load_jsonl(split_path)
            n = int(size)
            seq_len = n * n
            for row in rows:
                puzzle = row["puzzle"][:seq_len]
                solution = row["solution"][:seq_len]
                x0_rows = []
                src_rows = []
                for r in range(n):
                    puzzle_line = puzzle[r * n : (r + 1) * n]
                    line = solution[r * n : (r + 1) * n]
                    q_col = line.find("Q")
                    if q_col < 0:
                        raise ValueError(f"Invalid nqueens solution row without Q: n={n}, row={r}")
                    x0_rows.append(q_col)
                    if "Q" in puzzle_line:
                        src_rows.append(1)
                    else:
                        src_rows.append(0)
                unknown_rows = sum(1 for v in src_rows if v == 0)
                diff = row.get("difficulty", "easy")
                self.samples.append((x0_rows, src_rows, n, diff))
                base_w = difficulty_weight(diff, easy_weight, medium_weight, hard_weight)
                unknown_ratio = unknown_rows / float(max(1, n))
                self.sample_weights.append(base_w * (1.0 + unknown_ratio_weight * unknown_ratio))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int], int, str]:
        return self.samples[idx]


def collate_sudoku(batch):
    bsz = len(batch)
    x0 = torch.zeros((bsz, SUDOKU_MAX_LEN), dtype=torch.long)
    src_mask = torch.zeros((bsz, SUDOKU_MAX_LEN), dtype=torch.bool)
    valid_mask = torch.zeros((bsz, SUDOKU_MAX_LEN), dtype=torch.float32)
    n_vec = torch.zeros((bsz,), dtype=torch.long)
    for i, (x0i, srci, seq_len, n, _diff) in enumerate(batch):
        x0[i, :seq_len] = torch.tensor(x0i, dtype=torch.long)
        src_mask[i, :seq_len] = torch.tensor(srci, dtype=torch.bool)
        valid_mask[i, :seq_len] = 1.0
        n_vec[i] = n
    return x0, src_mask, valid_mask, n_vec


def collate_nqueens(batch):
    bsz = len(batch)
    x0 = torch.zeros((bsz, NQUEENS_MAX_N), dtype=torch.long)
    src_mask = torch.zeros((bsz, NQUEENS_MAX_N), dtype=torch.bool)
    valid_mask = torch.zeros((bsz, NQUEENS_MAX_N), dtype=torch.float32)
    n_vec = torch.zeros((bsz,), dtype=torch.long)
    for i, (x0_rows, src_rows, n, _diff) in enumerate(batch):
        x0[i, :n] = torch.tensor(x0_rows, dtype=torch.long)
        src_mask[i, :n] = torch.tensor(src_rows, dtype=torch.bool)
        valid_mask[i, :n] = 1.0
        n_vec[i] = n
    return x0, src_mask, valid_mask, n_vec


def build_transformer_encoder(layer: nn.TransformerEncoderLayer, num_layers: int) -> nn.TransformerEncoder:
    # Keep fixed padded sequence length in eval; nested-tensor optimization can shrink length.
    try:
        return nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
    except TypeError:
        return nn.TransformerEncoder(layer, num_layers=num_layers)


class SudokuMDMNet(nn.Module):
    def __init__(
        self,
        hidden_size: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        diffusion_steps: int = 20,
    ):
        super().__init__()
        self.max_len = SUDOKU_MAX_LEN
        self.input_vocab_size = SUDOKU_MAX_SYMBOL + 2  # 0..25 + [MASK]
        self.output_vocab_size = SUDOKU_MAX_SYMBOL + 1  # 0..25
        self.token_emb = nn.Embedding(self.input_vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(self.max_len, hidden_size)
        self.t_emb = nn.Embedding(diffusion_steps, hidden_size)
        self.n_emb = nn.Embedding(SUDOKU_MAX_SYMBOL + 1, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = build_transformer_encoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_size)
        self.out_proj = nn.Linear(hidden_size, self.output_vocab_size)

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, valid_mask: torch.Tensor, n_vec: torch.Tensor
    ) -> torch.Tensor:
        bsz, seq_len = x_t.shape
        pos = torch.arange(seq_len, device=x_t.device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
        t_emb = self.t_emb(t).unsqueeze(1)
        n_emb = self.n_emb(n_vec.clamp(min=0, max=SUDOKU_MAX_SYMBOL)).unsqueeze(1)
        h = self.token_emb(x_t) + self.pos_emb(pos) + t_emb + n_emb
        key_padding_mask = ~valid_mask.bool()
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        h = self.norm(h)
        return self.out_proj(h)


class NQueensMDMNet(nn.Module):
    def __init__(
        self,
        hidden_size: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        diffusion_steps: int = 20,
    ):
        super().__init__()
        self.max_len = NQUEENS_MAX_N
        self.max_n = NQUEENS_MAX_N
        self.input_vocab_size = NQUEENS_MAX_N + 1  # 0..31 + [MASK]
        self.output_vocab_size = NQUEENS_MAX_N  # 0..31
        self.token_emb = nn.Embedding(self.input_vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(self.max_len, hidden_size)
        self.t_emb = nn.Embedding(diffusion_steps, hidden_size)
        self.n_emb = nn.Embedding(self.max_n + 1, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = build_transformer_encoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_size)
        self.out_proj = nn.Linear(hidden_size, self.output_vocab_size)

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, valid_mask: torch.Tensor, n_vec: torch.Tensor
    ) -> torch.Tensor:
        bsz, seq_len = x_t.shape
        pos = torch.arange(seq_len, device=x_t.device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
        t_emb = self.t_emb(t).unsqueeze(1)
        n_emb = self.n_emb(n_vec.clamp(min=0, max=self.max_n)).unsqueeze(1)
        h = self.token_emb(x_t) + self.pos_emb(pos) + t_emb + n_emb
        key_padding_mask = ~valid_mask.bool()
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        h = self.norm(h)
        return self.out_proj(h)


def q_sample(
    x0: torch.Tensor,
    t: torch.Tensor,
    src_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    diffusion_steps: int,
    mask_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Match train-mdm style masking: mask each target position with prob (t+1)/T."""
    u = torch.rand_like(x0, dtype=torch.float)
    p = ((t + 1).float() / float(diffusion_steps)).unsqueeze(1)
    loss_mask = (u < p) & (~src_mask) & valid_mask.bool()
    x_t = x0.clone()
    x_t[loss_mask] = mask_token_id
    return x_t, loss_mask


def mdm_weighted_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    t: torch.Tensor,
    diffusion_steps: int,
    token_reweighting: bool,
    alpha: float,
    gamma: float,
    time_reweighting: str,
) -> torch.Tensor:
    vocab = logits.size(-1)
    loss = F.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1), reduction="none").float()
    flat_mask = loss_mask.reshape(-1)
    loss = loss.masked_fill(~flat_mask, 0.0)

    if token_reweighting:
        loss = alpha * (1.0 - torch.exp(-loss)).pow(gamma) * loss

    if time_reweighting == "original":
        time_w = (1.0 / (t + 1).float()).unsqueeze(1)
    elif time_reweighting == "linear":
        time_w = (diffusion_steps - t).float().unsqueeze(1)
    else:
        time_w = torch.ones((targets.size(0), 1), device=targets.device, dtype=torch.float32)

    time_w = time_w.expand_as(loss_mask).reshape(-1)
    denom = flat_mask.sum().clamp_min(1)
    return (loss * time_w).sum() / denom


def create_device(gpus_arg: str) -> Tuple[torch.device, int]:
    if gpus_arg and gpus_arg != "auto":
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus_arg
    if torch.cuda.is_available():
        device = torch.device("cuda")
        return device, torch.cuda.device_count()
    return torch.device("cpu"), 0


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    total_steps: int,
    warmup_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, min(total_steps - 1, int(total_steps * warmup_ratio)))

    def lr_lambda(current_step: int) -> float:
        if scheduler_type == "constant":
            return 1.0
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def unwrap_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return model.module.state_dict() if hasattr(model, "module") else model.state_dict()


def evaluate_gen_sudoku(
    model: nn.Module, loader: DataLoader, device: torch.device, diffusion_steps: int
) -> Dict:
    model.eval()
    total_tokens = 0
    correct_top1 = 0
    correct_top3 = 0
    total_puzzles = 0
    exact_puzzles = 0
    by_size = {}
    with torch.no_grad():
        for x0, src_mask, valid_mask, n_vec in loader:
            x0 = x0.to(device, non_blocking=True)
            src_mask = src_mask.to(device, non_blocking=True)
            valid_mask = valid_mask.to(device, non_blocking=True)
            n_vec = n_vec.to(device, non_blocking=True)
            bsz = x0.size(0)

            # Inference aligned with train-mdm: all unknowns masked at final timestep.
            x_t = x0.clone()
            unknown = (~src_mask) & valid_mask.bool()
            x_t[unknown] = SUDOKU_MASK_TOKEN
            t = torch.full((bsz,), diffusion_steps - 1, dtype=torch.long, device=device)
            logits = model(x_t, t, valid_mask, n_vec)
            seq_len = logits.size(1)
            x0_eval = x0[:, :seq_len]
            valid = unknown[:, :seq_len]
            pred = logits.argmax(dim=-1)
            topk = logits.topk(k=min(3, logits.size(-1)), dim=-1).indices

            total_tokens += int(valid.sum().item())
            correct_top1 += int(((pred == x0_eval) & valid).sum().item())
            correct_top3 += int(((topk == x0_eval.unsqueeze(-1)) & valid.unsqueeze(-1)).any(dim=-1).sum().item())

            for i in range(x0.size(0)):
                valid_i = valid[i]
                if not bool(valid_i.any()):
                    continue
                total_puzzles += 1
                n = int(n_vec[i].item())
                key = f"{n}x{n}"
                if key not in by_size:
                    by_size[key] = {"puzzles": 0, "exact": 0, "tokens": 0, "top1": 0, "top3": 0}
                by_size[key]["puzzles"] += 1
                by_size[key]["tokens"] += int(valid_i.sum().item())
                by_size[key]["top1"] += int(((pred[i] == x0_eval[i]) & valid_i).sum().item())
                by_size[key]["top3"] += int(
                    ((topk[i] == x0_eval[i].unsqueeze(-1)) & valid_i.unsqueeze(-1)).any(dim=-1).sum().item()
                )
                exact = bool((pred[i][valid_i] == x0_eval[i][valid_i]).all().item())
                if exact:
                    exact_puzzles += 1
                    by_size[key]["exact"] += 1

    out = {
        "unknown_token_acc": correct_top1 / max(1, total_tokens),
        "unknown_token_top3_acc": correct_top3 / max(1, total_tokens),
        "unknown_exact_puzzle_acc": exact_puzzles / max(1, total_puzzles),
        "unknown_tokens": total_tokens,
        "unknown_puzzles": total_puzzles,
        "by_size": {},
    }
    for key, s in sorted(by_size.items()):
        out["by_size"][key] = {
            "unknown_token_acc": s["top1"] / max(1, s["tokens"]),
            "unknown_token_top3_acc": s["top3"] / max(1, s["tokens"]),
            "unknown_exact_puzzle_acc": s["exact"] / max(1, s["puzzles"]),
            "unknown_tokens": s["tokens"],
            "unknown_puzzles": s["puzzles"],
        }
    return out


def evaluate_nqueens(
    model: nn.Module, loader: DataLoader, device: torch.device, diffusion_steps: int
) -> Dict:
    model.eval()
    total_rows = 0
    correct_rows = 0
    total_puzzles = 0
    exact_puzzles = 0
    by_size = {}
    with torch.no_grad():
        for x0, src_mask, valid_mask, n_vec in loader:
            x0 = x0.to(device, non_blocking=True)
            src_mask = src_mask.to(device, non_blocking=True)
            valid_mask = valid_mask.to(device, non_blocking=True)
            n_vec = n_vec.to(device, non_blocking=True)
            bsz = x0.size(0)
            x_t = x0.clone()
            unknown = (~src_mask) & valid_mask.bool()
            x_t[unknown] = NQUEENS_MASK_TOKEN
            t = torch.full((bsz,), diffusion_steps - 1, dtype=torch.long, device=device)
            logits = model(x_t, t, valid_mask, n_vec)
            seq_len = logits.size(1)
            x0_eval = x0[:, :seq_len]
            valid = unknown[:, :seq_len]
            pred = logits.argmax(dim=-1)
            total_rows += int(valid.sum().item())
            correct_rows += int(((pred == x0_eval) & valid).sum().item())
            for i in range(x0.size(0)):
                valid_i = valid[i]
                if not bool(valid_i.any()):
                    continue
                total_puzzles += 1
                n = int(n_vec[i].item())
                key = str(n)
                if key not in by_size:
                    by_size[key] = {"puzzles": 0, "exact": 0, "rows": 0, "correct_rows": 0}
                by_size[key]["puzzles"] += 1
                by_size[key]["rows"] += int(valid_i.sum().item())
                by_size[key]["correct_rows"] += int(((pred[i] == x0_eval[i]) & valid_i).sum().item())
                exact = bool((pred[i][valid_i] == x0_eval[i][valid_i]).all().item())
                if exact:
                    exact_puzzles += 1
                    by_size[key]["exact"] += 1
    out = {
        "unknown_row_acc": correct_rows / max(1, total_rows),
        "unknown_exact_puzzle_acc": exact_puzzles / max(1, total_puzzles),
        "unknown_rows": total_rows,
        "unknown_puzzles": total_puzzles,
        "by_size": {},
    }
    for key, s in sorted(by_size.items(), key=lambda kv: int(kv[0])):
        out["by_size"][key] = {
            "unknown_row_acc": s["correct_rows"] / max(1, s["rows"]),
            "unknown_exact_puzzle_acc": s["exact"] / max(1, s["puzzles"]),
            "unknown_rows": s["rows"],
            "unknown_puzzles": s["puzzles"],
        }
    return out


def train_gen_sudoku(
    data_root: Path,
    sizes: List[str],
    device: torch.device,
    gpu_count: int,
    epochs: int,
    batch_size: int,
    lr: float,
    num_workers: int,
    weighted_sampling: bool,
    easy_weight: float,
    medium_weight: float,
    hard_weight: float,
    unknown_ratio_weight: float,
    eval_split: str,
    eval_batch_size: int,
    eval_every: int,
    hidden_size: int,
    num_layers: int,
    num_heads: int,
    dropout: float,
    diffusion_steps: int,
    token_reweighting: bool,
    loss_alpha: float,
    loss_gamma: float,
    time_reweighting: str,
    lr_scheduler: str,
    warmup_ratio: float,
    grad_acc_steps: int,
    max_grad_norm: float,
) -> Dict:
    train_dataset = SudokuTrainDataset(
        data_root=data_root,
        sizes=sizes,
        split="train",
        easy_weight=easy_weight,
        medium_weight=medium_weight,
        hard_weight=hard_weight,
        unknown_ratio_weight=unknown_ratio_weight,
    )
    if len(train_dataset) == 0:
        raise RuntimeError("No generalized sudoku training rows found.")

    sampler = None
    if weighted_sampling:
        sampler = WeightedRandomSampler(
            weights=torch.tensor(train_dataset.sample_weights, dtype=torch.double),
            num_samples=len(train_dataset),
            replacement=True,
        )
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_sudoku,
    )

    eval_loader = None
    if eval_split != "none":
        eval_dataset = SudokuTrainDataset(
            data_root=data_root,
            sizes=sizes,
            split=eval_split,
            easy_weight=1.0,
            medium_weight=1.0,
            hard_weight=1.0,
            unknown_ratio_weight=0.0,
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_sudoku,
        )

    model = SudokuMDMNet(
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        diffusion_steps=diffusion_steps,
    ).to(device)
    if device.type == "cuda" and gpu_count > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    grad_acc_steps = max(1, grad_acc_steps)
    updates_per_epoch = (len(loader) + grad_acc_steps - 1) // grad_acc_steps
    scheduler = create_lr_scheduler(
        optimizer=optimizer,
        scheduler_type=lr_scheduler,
        total_steps=epochs * max(1, updates_per_epoch),
        warmup_ratio=warmup_ratio,
    )
    train_log = []
    eval_log = []
    best_metric = -1.0
    best_epoch = -1
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        update_steps = 0
        optimizer.zero_grad(set_to_none=True)
        for step_idx, (x0, src_mask, valid_mask, n_vec) in enumerate(loader, start=1):
            x0 = x0.to(device, non_blocking=True)
            src_mask = src_mask.to(device, non_blocking=True)
            valid_mask = valid_mask.to(device, non_blocking=True)
            n_vec = n_vec.to(device, non_blocking=True)
            bsz = x0.size(0)
            t = torch.randint(0, diffusion_steps, (bsz,), device=device)
            x_t, loss_mask = q_sample(
                x0=x0,
                t=t,
                src_mask=src_mask,
                valid_mask=valid_mask,
                diffusion_steps=diffusion_steps,
                mask_token_id=SUDOKU_MASK_TOKEN,
            )
            valid_tokens = int(loss_mask.sum().item())
            if valid_tokens <= 0:
                continue
            logits = model(x_t, t, valid_mask, n_vec)
            loss = mdm_weighted_loss(
                logits=logits,
                targets=x0,
                loss_mask=loss_mask,
                t=t,
                diffusion_steps=diffusion_steps,
                token_reweighting=token_reweighting,
                alpha=loss_alpha,
                gamma=loss_gamma,
                time_reweighting=time_reweighting,
            )
            (loss / grad_acc_steps).backward()
            do_step = (step_idx % grad_acc_steps == 0) or (step_idx == len(loader))
            if do_step:
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update_steps += 1
            total_loss += float(loss.item()) * valid_tokens
            total_tokens += valid_tokens
        avg_loss = total_loss / max(1, total_tokens)
        current_lr = float(optimizer.param_groups[0]["lr"])
        train_log.append(
            {"epoch": epoch, "loss": float(avg_loss), "tokens": int(total_tokens), "lr": current_lr, "updates": update_steps}
        )
        print(f"[gen_sudoku] epoch {epoch}/{epochs} loss={avg_loss:.6f} lr={current_lr:.6e} updates={update_steps}")

        if eval_loader is not None and (epoch % max(1, eval_every) == 0 or epoch == epochs):
            eval_metrics = evaluate_gen_sudoku(model, eval_loader, device, diffusion_steps=diffusion_steps)
            eval_metrics["epoch"] = epoch
            eval_log.append(eval_metrics)
            metric = float(eval_metrics["unknown_token_acc"])
            print(
                "[gen_sudoku] eval"
                f" token_acc={eval_metrics['unknown_token_acc']:.4f}"
                f" top3={eval_metrics['unknown_token_top3_acc']:.4f}"
                f" exact={eval_metrics['unknown_exact_puzzle_acc']:.4f}"
            )
            if metric > best_metric:
                best_metric = metric
                best_epoch = epoch
                best_state = copy.deepcopy(unwrap_state_dict(model))

    if best_state is not None:
        target = model.module if hasattr(model, "module") else model
        target.load_state_dict(best_state, strict=True)

    return {
        "model": model,
        "train_log": train_log,
        "eval_log": eval_log,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "num_samples": len(train_dataset),
    }


def train_nqueens(
    data_root: Path,
    sizes: List[str],
    device: torch.device,
    gpu_count: int,
    epochs: int,
    batch_size: int,
    lr: float,
    num_workers: int,
    weighted_sampling: bool,
    easy_weight: float,
    medium_weight: float,
    hard_weight: float,
    unknown_ratio_weight: float,
    eval_split: str,
    eval_batch_size: int,
    eval_every: int,
    hidden_size: int,
    num_layers: int,
    num_heads: int,
    dropout: float,
    diffusion_steps: int,
    token_reweighting: bool,
    loss_alpha: float,
    loss_gamma: float,
    time_reweighting: str,
    lr_scheduler: str,
    warmup_ratio: float,
    grad_acc_steps: int,
    max_grad_norm: float,
) -> Dict:
    train_dataset = NQueensTrainDataset(
        data_root=data_root,
        sizes=sizes,
        split="train",
        easy_weight=easy_weight,
        medium_weight=medium_weight,
        hard_weight=hard_weight,
        unknown_ratio_weight=unknown_ratio_weight,
    )
    if len(train_dataset) == 0:
        raise RuntimeError("No nqueens training rows found.")

    sampler = None
    if weighted_sampling:
        sampler = WeightedRandomSampler(
            weights=torch.tensor(train_dataset.sample_weights, dtype=torch.double),
            num_samples=len(train_dataset),
            replacement=True,
        )
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_nqueens,
    )

    eval_loader = None
    if eval_split != "none":
        eval_dataset = NQueensTrainDataset(
            data_root=data_root,
            sizes=sizes,
            split=eval_split,
            easy_weight=1.0,
            medium_weight=1.0,
            hard_weight=1.0,
            unknown_ratio_weight=0.0,
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_nqueens,
        )

    model = NQueensMDMNet(
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        diffusion_steps=diffusion_steps,
    ).to(device)
    if device.type == "cuda" and gpu_count > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    grad_acc_steps = max(1, grad_acc_steps)
    updates_per_epoch = (len(loader) + grad_acc_steps - 1) // grad_acc_steps
    scheduler = create_lr_scheduler(
        optimizer=optimizer,
        scheduler_type=lr_scheduler,
        total_steps=epochs * max(1, updates_per_epoch),
        warmup_ratio=warmup_ratio,
    )
    train_log = []
    eval_log = []
    best_metric = -1.0
    best_epoch = -1
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        update_steps = 0
        optimizer.zero_grad(set_to_none=True)
        for step_idx, (x0, src_mask, valid_mask, n_vec) in enumerate(loader, start=1):
            x0 = x0.to(device, non_blocking=True)
            src_mask = src_mask.to(device, non_blocking=True)
            valid_mask = valid_mask.to(device, non_blocking=True)
            n_vec = n_vec.to(device, non_blocking=True)
            bsz = x0.size(0)
            t = torch.randint(0, diffusion_steps, (bsz,), device=device)
            x_t, loss_mask = q_sample(
                x0=x0,
                t=t,
                src_mask=src_mask,
                valid_mask=valid_mask,
                diffusion_steps=diffusion_steps,
                mask_token_id=NQUEENS_MASK_TOKEN,
            )
            valid_rows = int(loss_mask.sum().item())
            if valid_rows <= 0:
                continue
            logits = model(x_t, t, valid_mask, n_vec)
            loss = mdm_weighted_loss(
                logits=logits,
                targets=x0,
                loss_mask=loss_mask,
                t=t,
                diffusion_steps=diffusion_steps,
                token_reweighting=token_reweighting,
                alpha=loss_alpha,
                gamma=loss_gamma,
                time_reweighting=time_reweighting,
            )
            (loss / grad_acc_steps).backward()
            do_step = (step_idx % grad_acc_steps == 0) or (step_idx == len(loader))
            if do_step:
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update_steps += 1
            total_loss += float(loss.item()) * valid_rows
            total_rows += valid_rows
        avg_loss = total_loss / max(1, total_rows)
        current_lr = float(optimizer.param_groups[0]["lr"])
        train_log.append(
            {"epoch": epoch, "loss": float(avg_loss), "rows": int(total_rows), "lr": current_lr, "updates": update_steps}
        )
        print(f"[nqueens] epoch {epoch}/{epochs} loss={avg_loss:.6f} lr={current_lr:.6e} updates={update_steps}")

        if eval_loader is not None and (epoch % max(1, eval_every) == 0 or epoch == epochs):
            eval_metrics = evaluate_nqueens(model, eval_loader, device, diffusion_steps=diffusion_steps)
            eval_metrics["epoch"] = epoch
            eval_log.append(eval_metrics)
            metric = float(eval_metrics["unknown_row_acc"])
            print(
                "[nqueens] eval"
                f" row_acc={eval_metrics['unknown_row_acc']:.4f}"
                f" exact={eval_metrics['unknown_exact_puzzle_acc']:.4f}"
            )
            if metric > best_metric:
                best_metric = metric
                best_epoch = epoch
                best_state = copy.deepcopy(unwrap_state_dict(model))

    if best_state is not None:
        target = model.module if hasattr(model, "module") else model
        target.load_state_dict(best_state, strict=True)

    return {
        "model": model,
        "train_log": train_log,
        "eval_log": eval_log,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "num_samples": len(train_dataset),
    }


def update_registry(registry_path: Path, family: str, checkpoint_path: Path, meta: Dict) -> None:
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    else:
        registry = {}
    registry[family] = {
        "checkpoint": str(checkpoint_path),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "format": "torch_family_v1",
        "meta": meta,
    }
    registry_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train extension family neural prior model.")
    parser.add_argument("--family", type=str, required=True, choices=["gen_sudoku", "nqueens"])
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--sizes", type=str, default="")
    parser.add_argument("--gpus", type=str, default="auto")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-weighted-sampling", action="store_true")
    parser.add_argument("--easy-weight", type=float, default=1.0)
    parser.add_argument("--medium-weight", type=float, default=1.5)
    parser.add_argument("--hard-weight", type=float, default=3.0)
    parser.add_argument("--unknown-ratio-weight", type=float, default=0.5)
    parser.add_argument("--eval-split", type=str, default="test", choices=["none", "test"])
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--diffusion-steps", type=int, default=20)
    parser.add_argument("--disable-token-reweighting", action="store_true")
    parser.add_argument("--loss-alpha", type=float, default=0.25)
    parser.add_argument("--loss-gamma", type=float, default=1.0)
    parser.add_argument("--time-reweighting", type=str, default="linear", choices=["none", "original", "linear"])
    parser.add_argument("--lr-scheduler", type=str, default="cosine", choices=["constant", "cosine"])
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    args = parser.parse_args()

    seed_everything(args.seed)
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    sizes = parse_sizes_arg(args.sizes, args.family)
    weighted_sampling = not args.disable_weighted_sampling
    token_reweighting = not args.disable_token_reweighting

    device, gpu_count = create_device(args.gpus)
    print(
        f"[train_family_model] family={args.family} sizes={sizes} device={device} "
        f"gpu_count={gpu_count} epochs={args.epochs} batch_size={args.batch_size}"
    )
    print(
        f"[train_family_model] weighted_sampling={weighted_sampling} "
        f"weights(e/m/h)=({args.easy_weight},{args.medium_weight},{args.hard_weight}) "
        f"unknown_ratio_weight={args.unknown_ratio_weight} eval_split={args.eval_split}"
    )
    print(
        f"[train_family_model] mdm diffusion_steps={args.diffusion_steps} "
        f"token_reweighting={token_reweighting} loss_alpha={args.loss_alpha} "
        f"loss_gamma={args.loss_gamma} time_reweighting={args.time_reweighting} "
        f"lr_scheduler={args.lr_scheduler} warmup_ratio={args.warmup_ratio} "
        f"grad_acc={args.gradient_accumulation_steps} max_grad_norm={args.max_grad_norm} "
        f"model(h={args.hidden_size},L={args.num_layers},H={args.num_heads},dropout={args.dropout})"
    )

    if args.family == "gen_sudoku":
        train_out = train_gen_sudoku(
            data_root=data_root,
            sizes=sizes,
            device=device,
            gpu_count=gpu_count,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            num_workers=args.num_workers,
            weighted_sampling=weighted_sampling,
            easy_weight=args.easy_weight,
            medium_weight=args.medium_weight,
            hard_weight=args.hard_weight,
            unknown_ratio_weight=args.unknown_ratio_weight,
            eval_split=args.eval_split,
            eval_batch_size=args.eval_batch_size,
            eval_every=args.eval_every,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            diffusion_steps=args.diffusion_steps,
            token_reweighting=token_reweighting,
            loss_alpha=args.loss_alpha,
            loss_gamma=args.loss_gamma,
            time_reweighting=args.time_reweighting,
            lr_scheduler=args.lr_scheduler,
            warmup_ratio=args.warmup_ratio,
            grad_acc_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
        )
        model_type = "SudokuMDMNet"
    else:
        train_out = train_nqueens(
            data_root=data_root,
            sizes=sizes,
            device=device,
            gpu_count=gpu_count,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            num_workers=args.num_workers,
            weighted_sampling=weighted_sampling,
            easy_weight=args.easy_weight,
            medium_weight=args.medium_weight,
            hard_weight=args.hard_weight,
            unknown_ratio_weight=args.unknown_ratio_weight,
            eval_split=args.eval_split,
            eval_batch_size=args.eval_batch_size,
            eval_every=args.eval_every,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            diffusion_steps=args.diffusion_steps,
            token_reweighting=token_reweighting,
            loss_alpha=args.loss_alpha,
            loss_gamma=args.loss_gamma,
            time_reweighting=args.time_reweighting,
            lr_scheduler=args.lr_scheduler,
            warmup_ratio=args.warmup_ratio,
            grad_acc_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
        )
        model_type = "NQueensMDMNet"

    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_path = output_root / f"{args.family}_{run_id}.pt"
    payload = {
        "family": args.family,
        "model_type": model_type,
        "sizes": sizes,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weighted_sampling": weighted_sampling,
        "mdm_config": {
            "diffusion_steps": args.diffusion_steps,
            "token_reweighting": token_reweighting,
            "loss_alpha": args.loss_alpha,
            "loss_gamma": args.loss_gamma,
            "time_reweighting": args.time_reweighting,
            "lr_scheduler": args.lr_scheduler,
            "warmup_ratio": args.warmup_ratio,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_grad_norm": args.max_grad_norm,
        },
        "model_config": {
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "dropout": args.dropout,
        },
        "weights": {
            "easy_weight": args.easy_weight,
            "medium_weight": args.medium_weight,
            "hard_weight": args.hard_weight,
            "unknown_ratio_weight": args.unknown_ratio_weight,
        },
        "eval_split": args.eval_split,
        "eval_every": args.eval_every,
        "state_dict": unwrap_state_dict(train_out["model"]),
        "train_log": train_out["train_log"],
        "eval_log": train_out["eval_log"],
        "best_metric": train_out["best_metric"],
        "best_epoch": train_out["best_epoch"],
    }
    torch.save(payload, out_path)

    registry_path = output_root / "checkpoints_registry.json"
    update_registry(
        registry_path=registry_path,
        family=args.family,
        checkpoint_path=out_path,
        meta={
            "family": args.family,
            "sizes": sizes,
            "data_root": str(data_root),
            "trainer": "train_family_model.py",
            "model_type": model_type,
            "num_samples": train_out["num_samples"],
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weighted_sampling": weighted_sampling,
            "mdm_config": {
                "diffusion_steps": args.diffusion_steps,
                "token_reweighting": token_reweighting,
                "loss_alpha": args.loss_alpha,
                "loss_gamma": args.loss_gamma,
                "time_reweighting": args.time_reweighting,
                "lr_scheduler": args.lr_scheduler,
                "warmup_ratio": args.warmup_ratio,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "max_grad_norm": args.max_grad_norm,
            },
            "model_config": {
                "hidden_size": args.hidden_size,
                "num_layers": args.num_layers,
                "num_heads": args.num_heads,
                "dropout": args.dropout,
            },
            "weights": {
                "easy_weight": args.easy_weight,
                "medium_weight": args.medium_weight,
                "hard_weight": args.hard_weight,
                "unknown_ratio_weight": args.unknown_ratio_weight,
            },
            "eval_split": args.eval_split,
            "eval_every": args.eval_every,
            "best_metric": train_out["best_metric"],
            "best_epoch": train_out["best_epoch"],
            "device": str(device),
            "gpu_count": gpu_count,
        },
    )

    metrics_path = output_root / f"{args.family}_{run_id}_train_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "family": args.family,
                "run_id": run_id,
                "num_samples": train_out["num_samples"],
                "train_log": train_out["train_log"],
                "eval_log": train_out["eval_log"],
                "best_metric": train_out["best_metric"],
                "best_epoch": train_out["best_epoch"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Family model saved: {out_path}")
    print(f"Train metrics    : {metrics_path}")
    print(f"Registry updated : {registry_path}")


if __name__ == "__main__":
    main()
