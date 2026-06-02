#!/usr/bin/env python3
"""Unified Table4 extension runner.

Runs both generalized Sudoku and N-Queens with:
- MRV+FC+LCV baseline
- DiBS-full (model prior + consistency + smart-call)
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "dataset" / "table4_extension"
DEFAULT_OUTPUT = PROJECT_ROOT / "DiBS" / "results" / "parallel" / "Table_4"
DEFAULT_REGISTRY = PROJECT_ROOT / "model" / "diffusion-vs-ar" / "output" / "extension" / "checkpoints_registry.json"
SUDOKU_MAX_LEN = 25 * 25
SUDOKU_MAX_SYMBOL = 25
SUDOKU_MASK_TOKEN = SUDOKU_MAX_SYMBOL + 1
NQUEENS_MAX_N = 32
NQUEENS_MASK_TOKEN = NQUEENS_MAX_N


@dataclass
class SmartCallConfig:
    min_interval: int = 5
    min_domain_threshold: int = 2


@dataclass
class InstanceResult:
    run_id: str
    task_family: str
    size: str
    solver: str
    instance_id: int
    puzzle: str
    status: str
    valid: bool
    time_ms: float
    nodes: int
    backtracks: int
    propagation_steps: int
    model_calls: int
    model_time_ms: float
    solution: Optional[str] = None
    error: str = ""


def build_transformer_encoder(layer: nn.TransformerEncoderLayer, num_layers: int) -> nn.TransformerEncoder:
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
        self.input_vocab_size = SUDOKU_MAX_SYMBOL + 2
        self.output_vocab_size = SUDOKU_MAX_SYMBOL + 1
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
        self.input_vocab_size = NQUEENS_MAX_N + 1
        self.output_vocab_size = NQUEENS_MAX_N
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


class LegacySudokuPriorNet(nn.Module):
    """Backward-compatible loader for old torch_family_v1 checkpoints."""

    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.max_len = SUDOKU_MAX_LEN
        self.vocab_size = SUDOKU_MAX_SYMBOL + 1
        self.token_emb = nn.Embedding(self.vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(self.max_len, hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, self.vocab_size),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = x.shape
        pos = torch.arange(seq_len, device=x.device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
        h = self.token_emb(x) + self.pos_emb(pos)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (h * mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / denom.unsqueeze(-1)
        pooled = pooled.expand(-1, seq_len, -1)
        return self.mlp(torch.cat([h, pooled], dim=-1))


class LegacyNQueensPriorNet(nn.Module):
    """Backward-compatible loader for old torch_family_v1 checkpoints."""

    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.max_len = NQUEENS_MAX_N * NQUEENS_MAX_N
        self.max_n = NQUEENS_MAX_N
        self.token_emb = nn.Embedding(2, hidden_size)
        self.pos_emb = nn.Embedding(self.max_len, hidden_size)
        self.row_emb = nn.Embedding(self.max_n, hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, self.max_n),
        )

    def forward(self, x: torch.Tensor, n_vec: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = x.shape
        pos = torch.arange(seq_len, device=x.device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
        h = self.token_emb(x) + self.pos_emb(pos)
        pooled = h.mean(dim=1)
        rows = torch.arange(self.max_n, device=x.device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
        row_h = self.row_emb(rows)
        pooled_rep = pooled.unsqueeze(1).expand(-1, self.max_n, -1)
        _ = n_vec
        return self.mlp(torch.cat([row_h, pooled_rep], dim=-1))


class ExtensionModel:
    """Task-family model loader supporting legacy json prior and torch checkpoints."""

    def __init__(self, registry_path: Path, device: str = "auto"):
        self.registry_path = registry_path
        self._cache: Dict[str, Dict] = {}
        self._index = self._load_registry()
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    def _load_registry(self) -> Dict:
        if not self.registry_path.exists():
            return {}
        return json.loads(self.registry_path.read_text(encoding="utf-8"))

    def _load_family(self, family_key: str) -> Optional[Dict]:
        if family_key in self._cache:
            return self._cache[family_key]
        item = self._index.get(family_key)
        if not item:
            return None
        checkpoint = Path(item["checkpoint"])
        if not checkpoint.exists():
            return None
        ckpt_format = item.get("format", "")
        if checkpoint.suffix == ".pt" or ckpt_format == "torch_family_v1":
            payload = torch.load(checkpoint, map_location=self.device)
            model_type = payload.get("model_type")
            mdm_conf = payload.get("mdm_config", {}) or {}
            model_conf = payload.get("model_config", {}) or {}
            diffusion_steps = int(mdm_conf.get("diffusion_steps", 20))
            hidden_size = int(model_conf.get("hidden_size", 512))
            num_layers = int(model_conf.get("num_layers", 6))
            num_heads = int(model_conf.get("num_heads", 8))
            dropout = float(model_conf.get("dropout", 0.1))
            if model_type == "SudokuMDMNet":
                model = SudokuMDMNet(
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    dropout=dropout,
                    diffusion_steps=diffusion_steps,
                ).to(self.device)
            elif model_type == "NQueensMDMNet":
                model = NQueensMDMNet(
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    dropout=dropout,
                    diffusion_steps=diffusion_steps,
                ).to(self.device)
            elif model_type == "SudokuPriorNet":
                model = LegacySudokuPriorNet(hidden_size=256).to(self.device)
            elif model_type == "NQueensPriorNet":
                model = LegacyNQueensPriorNet(hidden_size=256).to(self.device)
            else:
                raise ValueError(f"Unknown torch family model_type: {model_type}")
            model.load_state_dict(payload["state_dict"], strict=True)
            model.eval()
            out = {"kind": "torch", "payload": payload, "model": model}
        else:
            legacy = json.loads(checkpoint.read_text(encoding="utf-8"))
            out = {"kind": "legacy", "payload": legacy, "model": None}
        self._cache[family_key] = out
        return out

    @staticmethod
    def _symbol_char_to_val(ch: str, n: int) -> int:
        if ch in ("0", "."):
            return 0
        if n <= 16:
            alphabet = "0123456789ABCDEFG"
            idx = alphabet.find(ch.upper())
            if idx >= 0:
                return idx
        if ch.isdigit():
            return int(ch)
        return ord(ch.upper()) - ord("A") + 1

    @staticmethod
    def _sudoku_tensor_from_grid(
        grid: List[List[int]], diffusion_steps: int, n: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = [v for row in grid for v in row]
        seq_len = len(flat)
        x_t = torch.zeros((1, SUDOKU_MAX_LEN), dtype=torch.long)
        valid_mask = torch.zeros((1, SUDOKU_MAX_LEN), dtype=torch.float32)
        for i, v in enumerate(flat):
            if v > 0:
                x_t[0, i] = int(v)
            else:
                x_t[0, i] = SUDOKU_MASK_TOKEN
        valid_mask[0, :seq_len] = 1.0
        t = torch.tensor([max(0, diffusion_steps - 1)], dtype=torch.long)
        n_vec = torch.tensor([n], dtype=torch.long)
        return x_t, t, valid_mask, n_vec

    @staticmethod
    def _nqueens_tensor_from_rows(
        rows: List[int], diffusion_steps: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n = len(rows)
        x_t = torch.zeros((1, NQUEENS_MAX_N), dtype=torch.long)
        valid_mask = torch.zeros((1, NQUEENS_MAX_N), dtype=torch.float32)
        for r in range(n):
            x_t[0, r] = int(rows[r]) if rows[r] >= 0 else NQUEENS_MASK_TOKEN
            valid_mask[0, r] = 1.0
        t = torch.tensor([max(0, diffusion_steps - 1)], dtype=torch.long)
        n_vec = torch.tensor([n], dtype=torch.long)
        return x_t, t, valid_mask, n_vec

    def sudoku_probs(self, size: str, grid: List[List[int]]) -> Optional[Dict[int, Dict[int, float]]]:
        model = self._load_family("gen_sudoku")
        if not model:
            return None
        n = int(size.split("x")[0])
        if model["kind"] == "legacy":
            info = model["payload"].get("sizes", {}).get(size)
            if not info:
                return None
            priors = {}
            for cell, pd in info.get("priors", {}).items():
                priors[int(cell)] = {self._symbol_char_to_val(k, n): float(v) for k, v in pd.items()}
            return priors

        payload = model["payload"]
        model_type = payload.get("model_type")
        diffusion_steps = int(payload.get("mdm_config", {}).get("diffusion_steps", 20))
        torch_model = model["model"]
        x_t, t, valid_mask, n_vec = self._sudoku_tensor_from_grid(grid, diffusion_steps=diffusion_steps, n=n)
        x_t = x_t.to(self.device)
        t = t.to(self.device)
        valid_mask = valid_mask.to(self.device)
        n_vec = n_vec.to(self.device)
        with torch.no_grad():
            if model_type == "SudokuPriorNet":
                seq_len = n * n
                x_legacy = torch.zeros((1, SUDOKU_MAX_LEN), dtype=torch.long, device=self.device)
                for i in range(seq_len):
                    r, c = divmod(i, n)
                    v = grid[r][c]
                    x_legacy[0, i] = int(v) if v > 0 else 0
                logits = torch_model(x_legacy, valid_mask)[0].cpu().numpy()
            else:
                logits = torch_model(x_t, t, valid_mask, n_vec)[0].cpu().numpy()
        probs: Dict[int, Dict[int, float]] = {}
        for idx in range(n * n):
            raw = logits[idx, 1 : n + 1]
            raw = raw - raw.max()
            exp = np.exp(raw)
            den = float(exp.sum())
            if den <= 0:
                den = 1.0
            probs[idx] = {d: float(exp[d - 1] / den) for d in range(1, n + 1)}
        return probs

    def nqueens_probs(self, size: str, rows: List[int]) -> Optional[Dict[int, Dict[int, float]]]:
        model = self._load_family("nqueens")
        if not model:
            return None
        if model["kind"] == "legacy":
            info = model["payload"].get("sizes", {}).get(size)
            if not info:
                return None
            priors = {}
            for row, pd in info.get("priors", {}).items():
                priors[int(row)] = {int(k): float(v) for k, v in pd.items()}
            return priors

        payload = model["payload"]
        model_type = payload.get("model_type")
        diffusion_steps = int(payload.get("mdm_config", {}).get("diffusion_steps", 20))
        torch_model = model["model"]
        x_t, t, valid_mask, n_vec = self._nqueens_tensor_from_rows(rows, diffusion_steps=diffusion_steps)
        x_t = x_t.to(self.device)
        t = t.to(self.device)
        valid_mask = valid_mask.to(self.device)
        n_vec = n_vec.to(self.device)
        with torch.no_grad():
            if model_type == "NQueensPriorNet":
                # Legacy model expects flattened board inputs.
                n = int(size)
                flat = []
                for r in range(n):
                    for c in range(n):
                        flat.append(1 if rows[r] == c else 0)
                x_legacy = torch.zeros((1, NQUEENS_MAX_N * NQUEENS_MAX_N), dtype=torch.long, device=self.device)
                x_legacy[0, : n * n] = torch.tensor(flat, dtype=torch.long, device=self.device)
                logits = torch_model(x_legacy, n_vec)[0].cpu().numpy()
            else:
                logits = torch_model(x_t, t, valid_mask, n_vec)[0].cpu().numpy()
        n = int(size)
        probs: Dict[int, Dict[int, float]] = {}
        for r in range(n):
            raw = logits[r, :n]
            raw = raw - raw.max()
            exp = np.exp(raw)
            den = float(exp.sum())
            if den <= 0:
                den = 1.0
            probs[r] = {c: float(exp[c] / den) for c in range(n)}
        return probs


class GeneralizedSudokuSolver:
    def __init__(
        self,
        size: str,
        use_dibs: bool,
        use_lcv: bool,
        model: Optional[ExtensionModel],
        alpha: float = 0.8,
        timeout_ms: float = 60000,
        max_nodes: int = 1000000,
        smart_cfg: Optional[SmartCallConfig] = None,
    ):
        self.size = size
        self.n = int(size.split("x")[0])
        self.box = int(math.sqrt(self.n))
        self.use_dibs = use_dibs
        self.use_lcv = use_lcv
        self.model = model
        self.alpha = alpha
        self.timeout_ms = timeout_ms
        self.max_nodes = max_nodes
        self.smart_cfg = smart_cfg or SmartCallConfig()

        self.nodes = 0
        self.backtracks = 0
        self.propagation_steps = 0
        self.model_calls = 0
        self.model_time_ms = 0.0
        self._nodes_since_call = 0
        self._cell_probs: Dict[int, Dict[int, float]] = {}
        self._start_ms = 0.0
        self._peer_cache = {}
        for r in range(self.n):
            for c in range(self.n):
                peers = set()
                for i in range(self.n):
                    if i != c:
                        peers.add((r, i))
                    if i != r:
                        peers.add((i, c))
                br = (r // self.box) * self.box
                bc = (c // self.box) * self.box
                for pr in range(br, br + self.box):
                    for pc in range(bc, bc + self.box):
                        if (pr, pc) != (r, c):
                            peers.add((pr, pc))
                self._peer_cache[(r, c)] = list(peers)

    def _char_to_val(self, c: str) -> int:
        if c == "0":
            return 0
        if self.n <= 16:
            alphabet = "0123456789ABCDEFG"
            idx = alphabet.find(c.upper())
            if idx >= 0:
                return idx
        if c.isdigit():
            return int(c)
        return ord(c.upper()) - ord("A") + 1

    def _val_to_char(self, v: int) -> str:
        if v == 0:
            return "0"
        if self.n <= 16:
            alphabet = "0123456789ABCDEFG"
            return alphabet[v]
        return chr(ord("A") + v - 1)

    def parse(self, puzzle: str) -> List[List[int]]:
        out = [[0] * self.n for _ in range(self.n)]
        for i, ch in enumerate(puzzle[: self.n * self.n]):
            r, c = divmod(i, self.n)
            out[r][c] = self._char_to_val(ch)
        return out

    def stringify(self, grid: List[List[int]]) -> str:
        return "".join(self._val_to_char(v) for row in grid for v in row)

    def get_candidates(self, grid: List[List[int]], row: int, col: int) -> List[int]:
        if grid[row][col] != 0:
            return []
        used = set()
        for i in range(self.n):
            used.add(grid[row][i])
            used.add(grid[i][col])
        br = (row // self.box) * self.box
        bc = (col // self.box) * self.box
        for i in range(br, br + self.box):
            for j in range(bc, bc + self.box):
                used.add(grid[i][j])
        return [v for v in range(1, self.n + 1) if v not in used]

    def find_mrv_cells_with_candidates(
        self, grid: List[List[int]]
    ) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], List[int]]]:
        best = []
        min_len = self.n + 1
        cand_map: Dict[Tuple[int, int], List[int]] = {}
        for r in range(self.n):
            for c in range(self.n):
                if grid[r][c] != 0:
                    continue
                cand = self.get_candidates(grid, r, c)
                cand_map[(r, c)] = cand
                if len(cand) == 0:
                    return [(r, c)], cand_map
                if len(cand) < min_len:
                    min_len = len(cand)
                    best = [(r, c)]
                elif len(cand) == min_len:
                    best.append((r, c))
        return best, cand_map

    def propagate(self, grid: List[List[int]]) -> bool:
        changed = True
        while changed:
            changed = False
            for r in range(self.n):
                for c in range(self.n):
                    if grid[r][c] != 0:
                        continue
                    cand = self.get_candidates(grid, r, c)
                    if not cand:
                        return False
                    if len(cand) == 1:
                        grid[r][c] = cand[0]
                        changed = True
                        self.propagation_steps += 1
        return True

    def _peers(self, row: int, col: int) -> List[Tuple[int, int]]:
        return self._peer_cache[(row, col)]

    def _domain_size(self, grid: List[List[int]], rc: Tuple[int, int]) -> int:
        return len(self.get_candidates(grid, rc[0], rc[1]))

    def _should_call_model(self, min_domain: int) -> bool:
        if not self.use_dibs:
            return False
        if not self.model:
            return False
        if self._nodes_since_call < self.smart_cfg.min_interval:
            return False
        return min_domain >= self.smart_cfg.min_domain_threshold

    def _call_model(self, grid: List[List[int]]) -> None:
        t0 = time.perf_counter()
        probs: Dict[int, Dict[int, float]] = {}
        family_probs = self.model.sudoku_probs(self.size, grid) if self.model else None
        for r in range(self.n):
            for c in range(self.n):
                if grid[r][c] != 0:
                    continue
                cell_idx = r * self.n + c
                cand = self.get_candidates(grid, r, c)
                if not cand:
                    continue
                prior = family_probs.get(cell_idx) if family_probs else None
                if not prior:
                    val = 1.0 / len(cand)
                    probs[cell_idx] = {d: val for d in cand}
                    continue
                raw = np.array([max(1e-12, float(prior.get(d, 0.0))) for d in cand], dtype=np.float64)
                raw = raw / raw.sum()
                probs[cell_idx] = {d: float(raw[i]) for i, d in enumerate(cand)}
        self._cell_probs = probs
        self.model_calls += 1
        self._nodes_since_call = 0
        self.model_time_ms += (time.perf_counter() - t0) * 1000

    def _select_cell(self, grid: List[List[int]], mrv_cells: List[Tuple[int, int]]) -> Tuple[int, int]:
        if not self._cell_probs:
            return mrv_cells[0]
        best = mrv_cells[0]
        best_entropy = float("inf")
        for r, c in mrv_cells:
            idx = r * self.n + c
            pd = self._cell_probs.get(idx)
            if not pd:
                continue
            ent = 0.0
            for p in pd.values():
                if p > 0:
                    ent -= p * math.log2(p)
            if ent < best_entropy:
                best_entropy = ent
                best = (r, c)
        return best

    def _consistency_score(self, grid: List[List[int]], row: int, col: int, digit: int) -> float:
        idx = row * self.n + col
        _ = idx
        vals = []
        for pr, pc in self._peers(row, col):
            if grid[pr][pc] != 0:
                continue
            peer_idx = pr * self.n + pc
            pd = self._cell_probs.get(peer_idx)
            if not pd:
                continue
            vals.append(1.0 - pd.get(digit, 0.0))
        if not vals:
            return 0.0
        return float(np.mean(vals))

    def _lcv_score(
        self,
        grid: List[List[int]],
        row: int,
        col: int,
        digit: int,
        peer_candidates: Optional[Dict[Tuple[int, int], List[int]]] = None,
    ) -> int:
        """Higher score = less constraining (candidate remains available to more peers)."""
        if peer_candidates is not None:
            return sum(1 for cand in peer_candidates.values() if digit in cand)
        score = 0
        for pr, pc in self._peers(row, col):
            if grid[pr][pc] != 0:
                continue
            cand = self.get_candidates(grid, pr, pc)
            if digit in cand:
                score += 1
        return score

    def _order_values(self, grid: List[List[int]], row: int, col: int, candidates: List[int]) -> List[int]:
        if self._cell_probs:
            idx = row * self.n + col
            pd = self._cell_probs.get(idx)
            if pd:
                probs = np.array([pd.get(d, 0.0) for d in candidates], dtype=np.float64)
                cons = np.array([self._consistency_score(grid, row, col, d) for d in candidates], dtype=np.float64)
                score = self.alpha * probs + (1.0 - self.alpha) * cons
                order = np.argsort(score)[::-1]
                return [candidates[i] for i in order]
        if self.use_lcv:
            peer_candidates: Dict[Tuple[int, int], List[int]] = {}
            for pr, pc in self._peers(row, col):
                if grid[pr][pc] != 0:
                    continue
                peer_candidates[(pr, pc)] = self.get_candidates(grid, pr, pc)
            return sorted(
                candidates,
                key=lambda d: self._lcv_score(grid, row, col, d, peer_candidates=peer_candidates),
                reverse=True,
            )
        return candidates

    def _is_timeout(self) -> bool:
        if self.timeout_ms <= 0:
            return False
        return (time.perf_counter() * 1000 - self._start_ms) > self.timeout_ms

    def _validate(self, grid: List[List[int]]) -> bool:
        expected = set(range(1, self.n + 1))
        for i in range(self.n):
            row = set(grid[i])
            col = set(grid[r][i] for r in range(self.n))
            if row != expected or col != expected:
                return False
        for br in range(0, self.n, self.box):
            for bc in range(0, self.n, self.box):
                box = set()
                for r in range(br, br + self.box):
                    for c in range(bc, bc + self.box):
                        box.add(grid[r][c])
                if box != expected:
                    return False
        return True

    def _search(self, grid: List[List[int]]) -> bool:
        if self.nodes >= self.max_nodes or self._is_timeout():
            return False
        self.nodes += 1
        self._nodes_since_call += 1

        mrv, cand_map = self.find_mrv_cells_with_candidates(grid)
        if not mrv:
            return True
        min_domain = min(len(cand_map.get(rc, [])) for rc in mrv)
        if min_domain <= 0:
            return False

        if self._should_call_model(min_domain):
            self._call_model(grid)

        if self.use_dibs and self._cell_probs:
            r, c = self._select_cell(grid, mrv)
        else:
            r, c = mrv[0]
        cand = cand_map.get((r, c))
        if cand is None:
            cand = self.get_candidates(grid, r, c)
        if not cand:
            return False
        cand = self._order_values(grid, r, c, cand) if (self.use_dibs or self.use_lcv) else cand

        for d in cand:
            if self._is_timeout():
                return False
            snapshot = [row[:] for row in grid]
            grid[r][c] = d
            valid = self.propagate(grid)
            if valid and self._search(grid):
                return True
            grid[:] = snapshot
            self.backtracks += 1
        return False

    def solve(self, puzzle: str) -> Tuple[Optional[str], Dict]:
        grid = self.parse(puzzle)
        self.nodes = 0
        self.backtracks = 0
        self.propagation_steps = 0
        self.model_calls = 0
        self.model_time_ms = 0.0
        self._nodes_since_call = 0
        self._cell_probs = {}
        self._start_ms = time.perf_counter() * 1000

        if not self.propagate(grid):
            return None, {"solved": False, "valid": False}
        solved = self._search(grid)
        if solved:
            return self.stringify(grid), {"solved": True, "valid": self._validate(grid)}
        return None, {"solved": False, "valid": False}


class NQueensSolver:
    def __init__(
        self,
        size: str,
        use_dibs: bool,
        use_lcv: bool,
        model: Optional[ExtensionModel],
        alpha: float = 0.8,
        timeout_ms: float = 60000,
        max_nodes: int = 1000000,
        smart_cfg: Optional[SmartCallConfig] = None,
    ):
        self.size = size
        self.n = int(size)
        self.use_dibs = use_dibs
        self.use_lcv = use_lcv
        self.model = model
        self.alpha = alpha
        self.timeout_ms = timeout_ms
        self.max_nodes = max_nodes
        self.smart_cfg = smart_cfg or SmartCallConfig(min_interval=3, min_domain_threshold=2)

        self.nodes = 0
        self.backtracks = 0
        self.propagation_steps = 0
        self.model_calls = 0
        self.model_time_ms = 0.0
        self._nodes_since_call = 0
        self._row_probs: Dict[int, Dict[int, float]] = {}
        self._start_ms = 0.0

    def parse(self, puzzle: str) -> List[int]:
        rows = [-1] * self.n
        for r in range(self.n):
            line = puzzle[r * self.n : (r + 1) * self.n]
            q = line.find("Q")
            if q >= 0:
                rows[r] = q
        return rows

    def stringify(self, rows: List[int]) -> str:
        out = []
        for r in range(self.n):
            for c in range(self.n):
                out.append("Q" if rows[r] == c else ".")
        return "".join(out)

    def _is_safe(self, rows: List[int], row: int, col: int) -> bool:
        for r in range(self.n):
            c = rows[r]
            if c < 0 or r == row:
                continue
            if c == col:
                return False
            if abs(r - row) == abs(c - col):
                return False
        return True

    def _domain(self, rows: List[int], row: int) -> List[int]:
        if rows[row] >= 0:
            return [rows[row]]
        return [c for c in range(self.n) if self._is_safe(rows, row, c)]

    def _mrv_rows_with_domains(self, rows: List[int]) -> Tuple[List[int], Dict[int, List[int]]]:
        cand_rows = []
        min_domain = self.n + 1
        dom_map: Dict[int, List[int]] = {}
        for r in range(self.n):
            if rows[r] >= 0:
                continue
            dom = self._domain(rows, r)
            dom_map[r] = dom
            if not dom:
                return [r], dom_map
            if len(dom) < min_domain:
                min_domain = len(dom)
                cand_rows = [r]
            elif len(dom) == min_domain:
                cand_rows.append(r)
        return cand_rows, dom_map

    def _propagate(self, rows: List[int]) -> bool:
        changed = True
        while changed:
            changed = False
            for r in range(self.n):
                if rows[r] >= 0:
                    continue
                dom = self._domain(rows, r)
                if not dom:
                    return False
                if len(dom) == 1:
                    rows[r] = dom[0]
                    self.propagation_steps += 1
                    changed = True
        return True

    def _should_call_model(self, min_dom: int) -> bool:
        if not self.use_dibs or not self.model:
            return False
        if self._nodes_since_call < self.smart_cfg.min_interval:
            return False
        return min_dom >= self.smart_cfg.min_domain_threshold

    def _call_model(self, rows: List[int]) -> None:
        t0 = time.perf_counter()
        probs: Dict[int, Dict[int, float]] = {}
        family_probs = self.model.nqueens_probs(self.size, rows) if self.model else None
        for r in range(self.n):
            if rows[r] >= 0:
                continue
            dom = self._domain(rows, r)
            if not dom:
                continue
            prior = family_probs.get(r) if family_probs else None
            if not prior:
                p = 1.0 / len(dom)
                probs[r] = {c: p for c in dom}
                continue
            raw = np.array([max(1e-12, float(prior.get(c, 0.0))) for c in dom], dtype=np.float64)
            raw = raw / raw.sum()
            probs[r] = {dom[i]: float(raw[i]) for i in range(len(dom))}
        self._row_probs = probs
        self._nodes_since_call = 0
        self.model_calls += 1
        self.model_time_ms += (time.perf_counter() - t0) * 1000

    def _select_row(self, rows: List[int], mrv_rows: List[int]) -> int:
        if not self._row_probs:
            return mrv_rows[0]
        best = mrv_rows[0]
        best_entropy = float("inf")
        for r in mrv_rows:
            pd = self._row_probs.get(r)
            if not pd:
                continue
            ent = 0.0
            for p in pd.values():
                if p > 0:
                    ent -= p * math.log2(p)
            if ent < best_entropy:
                best_entropy = ent
                best = r
        return best

    def _conflict_cols(self, row: int, col: int, other_row: int) -> List[int]:
        delta = abs(other_row - row)
        cols = {col, col - delta, col + delta}
        return [c for c in cols if 0 <= c < self.n]

    def _consistency_score(self, rows: List[int], row: int, col: int) -> float:
        vals = []
        for r2 in range(self.n):
            if r2 == row or rows[r2] >= 0:
                continue
            pd = self._row_probs.get(r2)
            if not pd:
                continue
            conflict = self._conflict_cols(row, col, r2)
            conflict_prob = sum(pd.get(c, 0.0) for c in conflict)
            vals.append(1.0 - conflict_prob)
        if not vals:
            return 0.0
        return float(np.mean(vals))

    def _lcv_score(self, rows: List[int], row: int, col: int) -> int:
        score = 0
        snapshot = rows[row]
        rows[row] = col
        for r2 in range(self.n):
            if r2 == row or rows[r2] >= 0:
                continue
            score += len(self._domain(rows, r2))
        rows[row] = snapshot
        return score

    def _order_cols(self, rows: List[int], row: int, dom: List[int]) -> List[int]:
        if self._row_probs:
            pd = self._row_probs.get(row)
            if pd:
                probs = np.array([pd.get(c, 0.0) for c in dom], dtype=np.float64)
                cons = np.array([self._consistency_score(rows, row, c) for c in dom], dtype=np.float64)
                score = self.alpha * probs + (1.0 - self.alpha) * cons
                order = np.argsort(score)[::-1]
                return [dom[i] for i in order]
        if self.use_lcv:
            return sorted(dom, key=lambda c: self._lcv_score(rows, row, c), reverse=True)
        return dom

    def _is_timeout(self) -> bool:
        if self.timeout_ms <= 0:
            return False
        return (time.perf_counter() * 1000 - self._start_ms) > self.timeout_ms

    def _search(self, rows: List[int]) -> bool:
        if self.nodes >= self.max_nodes or self._is_timeout():
            return False
        self.nodes += 1
        self._nodes_since_call += 1

        mrv, dom_map = self._mrv_rows_with_domains(rows)
        if not mrv:
            return True
        min_dom = min(len(dom_map.get(r, [])) for r in mrv)
        if min_dom <= 0:
            return False

        if self._should_call_model(min_dom):
            self._call_model(rows)

        row = self._select_row(rows, mrv) if self.use_dibs else mrv[0]
        dom = dom_map.get(row)
        if dom is None:
            dom = self._domain(rows, row)
        if not dom:
            return False
        dom = self._order_cols(rows, row, dom) if (self.use_dibs or self.use_lcv) else dom

        for col in dom:
            if self._is_timeout():
                return False
            snapshot = rows[:]
            rows[row] = col
            valid = self._propagate(rows)
            if valid and self._search(rows):
                return True
            rows[:] = snapshot
            self.backtracks += 1
        return False

    def solve(self, puzzle: str) -> Tuple[Optional[str], Dict]:
        rows = self.parse(puzzle)
        self.nodes = 0
        self.backtracks = 0
        self.propagation_steps = 0
        self.model_calls = 0
        self.model_time_ms = 0.0
        self._nodes_since_call = 0
        self._row_probs = {}
        self._start_ms = time.perf_counter() * 1000

        if not self._propagate(rows):
            return None, {"solved": False, "valid": False}
        solved = self._search(rows)
        if solved:
            return self.stringify(rows), {"solved": True, "valid": True}
        return None, {"solved": False, "valid": False}


def load_records(path: Path, max_count: Optional[int]) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_count and len(rows) >= max_count:
                break
    return rows


def load_existing_instance_results(jsonl_path: Path) -> Dict[int, InstanceResult]:
    if not jsonl_path.exists():
        return {}
    out: Dict[int, InstanceResult] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                inst = InstanceResult(
                    run_id=row.get("run_id", ""),
                    task_family=row.get("task_family", ""),
                    size=row.get("size", ""),
                    solver=row.get("solver", ""),
                    instance_id=int(row.get("instance_id", -1)),
                    puzzle=row.get("puzzle", ""),
                    status=row.get("status", "error"),
                    valid=bool(row.get("valid", False)),
                    time_ms=float(row.get("time_ms", 0.0)),
                    nodes=int(row.get("nodes", 0)),
                    backtracks=int(row.get("backtracks", 0)),
                    propagation_steps=int(row.get("propagation_steps", 0)),
                    model_calls=int(row.get("model_calls", 0)),
                    model_time_ms=float(row.get("model_time_ms", 0.0)),
                    solution=row.get("solution"),
                    error=row.get("error", ""),
                )
                if inst.instance_id >= 0:
                    out[inst.instance_id] = inst
            except Exception:
                # Ignore malformed partial lines.
                continue
    return out


def is_solver_complete(existing: Dict[int, InstanceResult], total: int) -> bool:
    if total <= 0:
        return True
    if len(existing) < total:
        return False
    for i in range(total):
        if i not in existing:
            return False
    return True


def summarize(results: List[InstanceResult]) -> Dict:
    solved = [r for r in results if r.status == "solved"]
    total = len(results)
    times = [r.time_ms for r in results] if results else [0]
    nodes_all = [r.nodes for r in results] if results else [0]
    backs_all = [r.backtracks for r in results] if results else [0]
    model_calls_all = [r.model_calls for r in results] if results else [0]
    model_time_all = [r.model_time_ms for r in results] if results else [0]
    nodes_solved = [r.nodes for r in solved] if solved else [0]
    backs_solved = [r.backtracks for r in solved] if solved else [0]
    model_calls_solved = [r.model_calls for r in solved] if solved else [0]
    model_time_solved = [r.model_time_ms for r in solved] if solved else [0]
    timeout_count = sum(1 for r in results if r.status == "timeout")
    return {
        "total_puzzles": total,
        "solved_count": len(solved),
        "solved_pct": (len(solved) / total * 100.0) if total else 0.0,
        "timeout_count": timeout_count,
        "time_ms": {
            "mean": float(np.mean(times)),
            "median": float(np.median(times)),
            "p95": float(np.percentile(times, 95)),
        },
        "nodes": {"mean": float(np.mean(nodes_all)), "mean_solved_only": float(np.mean(nodes_solved))},
        "backtracks": {"mean": float(np.mean(backs_all)), "mean_solved_only": float(np.mean(backs_solved))},
        "model_calls": {"mean": float(np.mean(model_calls_all)), "mean_solved_only": float(np.mean(model_calls_solved))},
        "model_time_ms": {"mean": float(np.mean(model_time_all)), "mean_solved_only": float(np.mean(model_time_solved))},
    }


def write_tex_table(path: Path, rows: List[Dict]) -> None:
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Table 4 Extension: generalized Sudoku and N-Queens (MRV+FC+LCV vs DiBS-full).}")
    lines.append("\\label{tab:table4_extension}")
    lines.append("\\begin{tabular}{llrrrr}")
    lines.append("\\toprule")
    lines.append("Task & Solver & Solved\\% & Time Mean & Time P95 & Nodes \\\\")
    lines.append("\\midrule")
    for row in rows:
        lines.append(
            f"{row['task']} & {row['solver']} & {row['solved_pct']:.2f} & "
            f"{row['time_mean']:.1f} & {row['time_p95']:.1f} & {row['nodes_mean']:.0f} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    path.write_text("\n".join(lines), encoding="utf-8")


_WORKER_SOLVER_CACHE: Dict[Tuple, object] = {}


def _resolve_worker_device(model_device: str, gpu_id: Optional[int]) -> str:
    if model_device == "cpu":
        return "cpu"
    if model_device.startswith("cuda:"):
        return model_device
    if model_device == "cuda":
        return f"cuda:{gpu_id}" if gpu_id is not None else "cuda"
    if model_device == "auto":
        if torch.cuda.is_available():
            return f"cuda:{gpu_id}" if gpu_id is not None else "cuda"
        return "cpu"
    return model_device


def _get_or_create_solver(
    task_family: str,
    size: str,
    use_dibs: bool,
    registry: str,
    model_device: str,
    gpu_id: Optional[int],
    timeout_ms: int,
    max_nodes: int,
    alpha: float,
    smart_interval: int,
):
    key = (
        task_family,
        size,
        use_dibs,
        registry,
        model_device,
        gpu_id if gpu_id is not None else -1,
        timeout_ms,
        max_nodes,
        alpha,
        smart_interval,
    )
    if key in _WORKER_SOLVER_CACHE:
        return _WORKER_SOLVER_CACHE[key]

    model = None
    if use_dibs:
        resolved_device = _resolve_worker_device(model_device, gpu_id)
        model = ExtensionModel(Path(registry), device=resolved_device)
    if task_family == "generalized_sudoku":
        solver = GeneralizedSudokuSolver(
            size=size,
            use_dibs=use_dibs,
            use_lcv=(not use_dibs),
            model=model,
            timeout_ms=timeout_ms,
            max_nodes=max_nodes,
            alpha=alpha,
            smart_cfg=SmartCallConfig(min_interval=smart_interval, min_domain_threshold=2),
        )
    else:
        solver = NQueensSolver(
            size=size,
            use_dibs=use_dibs,
            use_lcv=(not use_dibs),
            model=model,
            timeout_ms=timeout_ms,
            max_nodes=max_nodes,
            alpha=alpha,
            smart_cfg=SmartCallConfig(min_interval=max(2, smart_interval // 2), min_domain_threshold=2),
        )
    _WORKER_SOLVER_CACHE[key] = solver
    return solver


def _solve_one_instance(
    run_id: str,
    task_family: str,
    size: str,
    solver_name: str,
    use_dibs: bool,
    puzzle: str,
    instance_id: int,
    registry: str,
    model_device: str,
    gpu_id: Optional[int],
    timeout_ms: int,
    max_nodes: int,
    alpha: float,
    smart_interval: int,
) -> InstanceResult:
    solver = _get_or_create_solver(
        task_family=task_family,
        size=size,
        use_dibs=use_dibs,
        registry=registry,
        model_device=model_device,
        gpu_id=gpu_id,
        timeout_ms=timeout_ms,
        max_nodes=max_nodes,
        alpha=alpha,
        smart_interval=smart_interval,
    )
    t0 = time.perf_counter()
    try:
        solution, metrics = solver.solve(puzzle)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        status = "solved" if metrics["solved"] else "failed"
        if timeout_ms > 0 and elapsed_ms > timeout_ms:
            status = "timeout"
        return InstanceResult(
            run_id=run_id,
            task_family=task_family,
            size=size,
            solver=solver_name,
            instance_id=instance_id,
            puzzle=puzzle,
            status=status,
            valid=bool(metrics["valid"]),
            time_ms=elapsed_ms,
            nodes=solver.nodes,
            backtracks=solver.backtracks,
            propagation_steps=solver.propagation_steps,
            model_calls=solver.model_calls,
            model_time_ms=solver.model_time_ms,
            solution=solution,
        )
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return InstanceResult(
            run_id=run_id,
            task_family=task_family,
            size=size,
            solver=solver_name,
            instance_id=instance_id,
            puzzle=puzzle,
            status="error",
            valid=False,
            time_ms=elapsed_ms,
            nodes=0,
            backtracks=0,
            propagation_steps=0,
            model_calls=0,
            model_time_ms=0.0,
            error=str(exc),
        )


def run_task(
    run_id: str,
    task_family: str,
    size: str,
    records: List[Dict],
    out_dir: Path,
    registry: str,
    model_device: str,
    timeout_ms: int,
    max_nodes: int,
    alpha: float,
    smart_interval: int,
    workers: int,
    gpus: Optional[List[int]],
    resume: bool,
) -> Tuple[Dict, Dict]:
    solver_specs = [
        ("MRV+FC+LCV", False),
        ("DiBS-full", True),
    ]
    summaries = {}
    per_solver_rows = []

    for solver_name, use_dibs in solver_specs:
        task_tag = f"{task_family}_{size}_{solver_name}".replace("+", "_")
        jsonl_path = out_dir / f"{run_id}_{task_tag}.jsonl"
        summary_path = out_dir / f"{run_id}_{task_tag}_summary.json"
        total = len(records)
        existing = load_existing_instance_results(jsonl_path) if resume else {}

        if resume and is_solver_complete(existing, total) and summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summaries[solver_name] = summary
            per_solver_rows.append(summary)
            print(
                f"[{task_family}/{size}] {solver_name}: resume-skip complete "
                f"({summary['solved_count']}/{summary['total_puzzles']} solved)",
                flush=True,
            )
            continue

        pending_indices = [i for i in range(total) if i not in existing]
        solved_so_far = sum(1 for r in existing.values() if r.status == "solved")
        timeout_so_far = sum(1 for r in existing.values() if r.status == "timeout")
        done_initial = len(existing)
        t_start = time.perf_counter()
        results_map: Dict[int, InstanceResult] = dict(existing)
        effective_workers = max(1, workers)
        if solver_name == "MRV+FC+LCV":
            effective_workers = max(1, workers)
        elif gpus:
            effective_workers = min(max(1, workers), len(gpus))
        print(
            f"[{task_family}/{size}] {solver_name}: start {total} puzzles "
            f"(resume_done={done_initial}, pending={len(pending_indices)}, "
            f"workers={effective_workers}, gpus={gpus if gpus else 'none'})",
            flush=True,
        )

        file_mode = "a" if (resume and jsonl_path.exists()) else "w"
        if effective_workers == 1:
            gpu_id = gpus[0] if (gpus and solver_name == "DiBS-full") else None
            with jsonl_path.open(file_mode, encoding="utf-8") as writer:
                for idx in pending_indices:
                    row = records[idx]
                    r = _solve_one_instance(
                        run_id=run_id,
                        task_family=task_family,
                        size=size,
                        solver_name=solver_name,
                        use_dibs=use_dibs,
                        puzzle=row["puzzle"],
                        instance_id=idx,
                        registry=registry,
                        model_device=model_device,
                        gpu_id=gpu_id,
                        timeout_ms=timeout_ms,
                        max_nodes=max_nodes,
                        alpha=alpha,
                        smart_interval=smart_interval,
                    )
                    results_map[r.instance_id] = r
                    writer.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
                    writer.flush()
                    if r.status == "solved":
                        solved_so_far += 1
                    elif r.status == "timeout":
                        timeout_so_far += 1
                    done = len(results_map)
                    if done % 10 == 0 or done == total:
                        elapsed = time.perf_counter() - t_start
                        processed = max(1, done - done_initial)
                        eta = (elapsed / processed) * (total - done) if done < total else 0.0
                        print(
                            f"[{task_family}/{size}] {solver_name}: {done}/{total} "
                            f"solved={solved_so_far} timeout={timeout_so_far} "
                            f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                            flush=True,
                        )
        else:
            tasks = []
            for idx in pending_indices:
                row = records[idx]
                gpu_id = None
                if solver_name == "DiBS-full" and gpus:
                    gpu_id = gpus[idx % len(gpus)]
                tasks.append(
                    (
                        run_id,
                        task_family,
                        size,
                        solver_name,
                        use_dibs,
                        row["puzzle"],
                        idx,
                        registry,
                        model_device,
                        gpu_id,
                        timeout_ms,
                        max_nodes,
                        alpha,
                        smart_interval,
                    )
                )

            mp_ctx = multiprocessing.get_context("spawn")
            with jsonl_path.open(file_mode, encoding="utf-8") as writer:
                with ProcessPoolExecutor(max_workers=effective_workers, mp_context=mp_ctx) as ex:
                    futures = {ex.submit(_solve_one_instance, *t): t[6] for t in tasks}
                    for fut in as_completed(futures):
                        r = fut.result()
                        results_map[r.instance_id] = r
                        writer.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
                        writer.flush()
                        if r.status == "solved":
                            solved_so_far += 1
                        elif r.status == "timeout":
                            timeout_so_far += 1
                        done = len(results_map)
                        if done % 10 == 0 or done == total:
                            elapsed = time.perf_counter() - t_start
                            processed = max(1, done - done_initial)
                            eta = (elapsed / processed) * (total - done) if done < total else 0.0
                            print(
                                f"[{task_family}/{size}] {solver_name}: {done}/{total} "
                                f"solved={solved_so_far} timeout={timeout_so_far} "
                                f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                                flush=True,
                            )

        results = [results_map[i] for i in sorted(results_map.keys()) if i < total]

        summary = summarize(results)
        summary.update({"task_family": task_family, "size": size, "solver": solver_name})
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        summaries[solver_name] = summary
        per_solver_rows.append(summary)
        print(
            f"[{task_family}/{size}] {solver_name}: solved={summary['solved_count']}/{summary['total_puzzles']} "
            f"({summary['solved_pct']:.2f}%), time_mean={summary['time_ms']['mean']:.1f}ms"
        )
    return summaries, {"rows": per_solver_rows}


def parse_task_filter(value: str, sudoku_sizes: List[str], nqueens_sizes: List[str]) -> List[Tuple[str, str]]:
    value = value.strip()
    if value == "all":
        return [("generalized_sudoku", s) for s in sudoku_sizes] + [("nqueens", s) for s in nqueens_sizes]
    out = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid task format '{item}'. Expected family:size")
        family, size = item.split(":", 1)
        out.append((family, size))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified Table4 extension runner.")
    parser.add_argument("--data-root", type=str, default=str(DATA_ROOT))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--registry", type=str, default=str(DEFAULT_REGISTRY))
    parser.add_argument("--tasks", type=str, default="all", help="all or csv family:size, e.g. generalized_sudoku:4x4,nqueens:8")
    parser.add_argument("--max-puzzles", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-ms", type=int, default=60000)
    parser.add_argument("--max-nodes", type=int, default=1000000)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--smart-interval", type=int, default=5)
    parser.add_argument("--model-device", type=str, default="auto", help="auto/cpu/cuda")
    parser.add_argument("--workers", type=int, default=1, help="parallel worker processes")
    parser.add_argument("--gpus", type=str, default="", help="GPU ids for DiBS workers, e.g. 0,1,2,3")
    parser.add_argument("--resume", action="store_true", help="resume from existing per-instance outputs")
    parser.add_argument("--run-id", type=str, default="", help="resume target run_id; empty means new run")
    parser.add_argument("--sudoku-sizes", type=str, default="4x4,16x16,25x25", help="used when tasks=all")
    parser.add_argument("--nqueens-sizes", type=str, default="8,9,10", help="used when tasks=all")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    run_id = args.run_id.strip() if args.run_id.strip() else time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    sudoku_sizes = [s.strip() for s in args.sudoku_sizes.split(",") if s.strip()]
    nqueens_sizes = [s.strip() for s in args.nqueens_sizes.split(",") if s.strip()]
    task_specs = parse_task_filter(args.tasks, sudoku_sizes, nqueens_sizes)
    gpu_list: List[int] = []
    if args.gpus.strip():
        gpu_list = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]

    meta = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_root": args.data_root,
        "registry": args.registry,
        "tasks": task_specs,
        "max_puzzles": args.max_puzzles,
        "seed": args.seed,
        "timeout_ms": args.timeout_ms,
        "max_nodes": args.max_nodes,
        "alpha": args.alpha,
        "smart_interval": args.smart_interval,
        "model_device": args.model_device,
        "workers": args.workers,
        "gpus": gpu_list,
        "resume": bool(args.resume),
        "sudoku_sizes": sudoku_sizes,
        "nqueens_sizes": nqueens_sizes,
    }
    meta_path = out_dir / f"{run_id}_meta.json"
    if not (args.resume and meta_path.exists()):
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    all_summaries = []
    tex_rows = []
    for family, size in task_specs:
        data_path = Path(args.data_root) / family / size / "test.jsonl"
        if not data_path.exists():
            print(f"Skip {family}/{size}: missing {data_path}")
            continue
        records = load_records(data_path, args.max_puzzles)
        if not records:
            print(f"Skip {family}/{size}: empty records")
            continue
        summaries, payload = run_task(
            run_id=run_id,
            task_family=family,
            size=size,
            records=records,
            out_dir=out_dir,
            registry=args.registry,
            model_device=args.model_device,
            timeout_ms=args.timeout_ms,
            max_nodes=args.max_nodes,
            alpha=args.alpha,
            smart_interval=args.smart_interval,
            workers=args.workers,
            gpus=gpu_list,
            resume=args.resume,
        )
        _ = payload
        all_summaries.extend(summaries.values())
        task_name = f"{family}:{size}"
        for solver_name in ("MRV+FC+LCV", "DiBS-full"):
            s = summaries[solver_name]
            tex_rows.append(
                {
                    "task": task_name,
                    "solver": solver_name,
                    "solved_pct": s["solved_pct"],
                    "time_mean": s["time_ms"]["mean"],
                    "time_p95": s["time_ms"]["p95"],
                    "nodes_mean": s["nodes"]["mean"],
                }
            )

    all_path = out_dir / f"{run_id}_all_summaries.json"
    all_path.write_text(json.dumps(all_summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    tex_path = out_dir / f"{run_id}_table4_extension.tex"
    write_tex_table(tex_path, tex_rows)
    print(f"All summaries: {all_path}")
    print(f"LaTeX table  : {tex_path}")


if __name__ == "__main__":
    main()
