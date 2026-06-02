from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Sequence, Tuple

import torch
from transformers import GPT2Config, GPT2LMHeadModel


class SATMDMWrapper:
    """Small Table5-only adapter from SAT partial assignments to MDM logits."""

    SEP = 1
    MASK = 2
    EOS = 3
    PAD = 0

    def __init__(self, checkpoint_path: str, task_name: str, device: str = "cuda") -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.task_name = task_name
        self.num_vars = int(task_name.replace("3sat", ""))
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
        self.model_max_length = 325
        self.last_inference_time_ms = 0.0
        self._load_tokenizer_config()
        self._load_model()

    def _load_tokenizer_config(self) -> None:
        cfg_path = self.checkpoint_path / "tokenizer_config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        self.vocab = list(cfg["vocab"])
        self.model_max_length = int(cfg.get("model_max_length", 325))
        self.token_to_id: Dict[str, int] = {
            "[PAD]": self.PAD,
            "[SEP]": self.SEP,
            "[MASK]": self.MASK,
            "[EOS]": self.EOS,
            "[UNK]": 4,
        }
        self.token_to_id.update({tok: i + 5 for i, tok in enumerate(self.vocab)})

    def _load_model(self) -> None:
        config = GPT2Config.from_json_file(str(self.checkpoint_path / "config.json"))
        model = GPT2LMHeadModel(config)
        state = torch.load(self.checkpoint_path / "pytorch_model.bin", map_location="cpu")
        if any(k.startswith("model.") for k in state):
            state = {k[6:]: v for k, v in state.items() if k.startswith("model.")}
        model.load_state_dict(state, strict=False)
        model.to(self.device)
        model.eval()
        self.model = model

    def _encode_chars(self, text: str) -> Sequence[int]:
        return [self.token_to_id.get(ch, self.token_to_id["[UNK]"]) for ch in text]

    def literal_tokens(self) -> Tuple[Sequence[str], Sequence[str]]:
        if self.num_vars == 5:
            return [str(i) for i in range(5)], [str(i) for i in range(5, 10)]
        return [str(i) for i in range(1, self.num_vars + 1)], [chr(ord("b") + i) for i in range(self.num_vars)]

    def _assignment_token_ids(self, assignment: Sequence[int]) -> Tuple[Sequence[int], Sequence[int]]:
        pos_tokens, neg_tokens = self.literal_tokens()
        token_ids = []
        var_positions = []
        for var in range(1, self.num_vars + 1):
            if var > 1:
                token_ids.append(self.token_to_id[","])
            var_positions.append(len(token_ids))
            val = assignment[var] if var < len(assignment) else 0
            if val > 0:
                token_ids.append(self.token_to_id[pos_tokens[var - 1]])
            elif val < 0:
                token_ids.append(self.token_to_id[neg_tokens[var - 1]])
            else:
                token_ids.append(self.MASK)
        return token_ids, var_positions

    def _forward_logits(self, raw_input: str, assignment: Sequence[int]):
        start = time.perf_counter()
        src_ids = list(self._encode_chars(raw_input))
        tgt_ids, var_positions = self._assignment_token_ids(assignment)
        input_ids = src_ids + [self.SEP] + list(tgt_ids) + [self.EOS]
        if len(input_ids) > self.model_max_length:
            input_ids = input_ids[: self.model_max_length]
        if len(input_ids) < self.model_max_length:
            input_ids = input_ids + [self.PAD] * (self.model_max_length - len(input_ids))

        ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(ids)
        with torch.no_grad():
            logits = self.model(input_ids=ids, attention_mask=attention_mask).logits
            logits = torch.cat([logits[:, 0:1], logits[:, :-1]], dim=1)
        self.last_inference_time_ms = (time.perf_counter() - start) * 1000.0
        return logits[0], len(src_ids), var_positions

    def score_literal_pair(self, raw_input: str, assignment: Sequence[int], var: int) -> Tuple[float, float]:
        logits, src_len, var_positions = self._forward_logits(raw_input, assignment)
        pos_tokens, neg_tokens = self.literal_tokens()
        target_pos = src_len + 1 + var_positions[var - 1]
        pos_id = self.token_to_id[pos_tokens[var - 1]]
        neg_id = self.token_to_id[neg_tokens[var - 1]]
        pair = logits[target_pos, [pos_id, neg_id]].float().cpu()
        return float(pair[0].item()), float(pair[1].item())

    def score_literals(self, raw_input: str, assignment: Sequence[int], literals: Sequence[int]) -> Dict[int, float]:
        logits, src_len, var_positions = self._forward_logits(raw_input, assignment)
        pos_tokens, neg_tokens = self.literal_tokens()
        out: Dict[int, float] = {}
        for lit in literals:
            var = abs(int(lit))
            if var < 1 or var > self.num_vars:
                continue
            target_pos = src_len + 1 + var_positions[var - 1]
            tok = pos_tokens[var - 1] if lit > 0 else neg_tokens[var - 1]
            out[int(lit)] = float(logits[target_pos, self.token_to_id[tok]].float().cpu().item())
        return out
