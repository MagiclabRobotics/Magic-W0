#!/usr/bin/env python
# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""FAST discrete action tokenizer (pi0-FAST / ``physical-intelligence/fast``).

Vendored, dependency-light copy of ``UniversalActionProcessor``
(``processing_action_tokenizer.py`` in the ``physical-intelligence/fast`` HF
repo), stripped of ``ProcessorMixin`` so it can be rebuilt fully offline from a
saved directory (``bpe_tokenizer/tokenizer.json`` + ``processor_config.json``)
without ``AutoProcessor``/``trust_remote_code``/network. The shared training
path uses it to turn a continuous action chunk into discrete
tokens (DCT along time -> round(coeff * scale) -> byte-level BPE).

The encode/decode math is byte-for-byte the original; only the construction and
loading are simplified. ``__call__`` returns, per sample, a list of BPE ids in
``[0, vocab_size)``; the training policy maps those onto reserved VLM vocabulary
ids and computes next-token cross-entropy.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.fft import dct, idct


class FastActionTokenizer:
    """Byte-level BPE over DCT-compressed action chunks (pi0-FAST).

    Args:
        bpe_tokenizer: a ``PreTrainedTokenizerFast`` over the FAST byte alphabet.
        scale: DCT coefficient quantization scale (``round(coeff * scale)``).
        vocab_size: BPE vocab size (number of distinct FAST token ids).
        min_token: byte offset subtracted before ``chr(...)`` encoding.
        action_dim / time_horizon: cached for ``decode`` (encode overrides them).
    """

    def __init__(
        self,
        bpe_tokenizer,
        scale: float = 10.0,
        vocab_size: int = 1024,
        min_token: int = 0,
        *,
        action_dim: int | None = None,
        time_horizon: int | None = None,
    ):
        self.bpe_tokenizer = bpe_tokenizer
        self.scale = scale
        self.vocab_size = vocab_size
        self.min_token = min_token
        self.time_horizon = time_horizon
        self.action_dim = action_dim
        self.called_time_horizon = time_horizon
        self.called_action_dim = action_dim

    def __call__(self, action_chunk: np.ndarray) -> list[list[int]]:
        assert action_chunk.ndim <= 3, (
            "Only 3 dimensions supported: [batch, timesteps, action_dim]"
        )
        if action_chunk.ndim == 2:
            action_chunk = action_chunk[None, ...]

        self.called_time_horizon = action_chunk.shape[-2]
        self.called_action_dim = action_chunk.shape[-1]

        dct_coeff = dct(action_chunk, axis=1, norm="ortho")
        dct_coeff = np.around(dct_coeff * self.scale)
        tokens = []
        for elem in dct_coeff:
            token_str = "".join(
                map(chr, np.maximum(elem.flatten() - self.min_token, 0).astype(int))
            )
            tokens.append(self.bpe_tokenizer(token_str)["input_ids"])
        return tokens

    def decode(
        self,
        tokens: list[list[int]],
        *,
        time_horizon: int | None = None,
        action_dim: int | None = None,
    ) -> np.ndarray:
        self.time_horizon = (
            time_horizon or self.time_horizon or self.called_time_horizon
        )
        self.action_dim = action_dim or self.action_dim or self.called_action_dim
        self.called_time_horizon = self.time_horizon
        self.called_action_dim = self.action_dim
        assert self.time_horizon is not None and self.action_dim is not None, (
            "Tokenizer not initialized, call encode() once or pass in time_horizon and action_dim."
        )

        decoded_actions = []
        for token in tokens:
            try:
                decoded_tokens = self.bpe_tokenizer.decode(token)
                decoded_dct_coeff = (
                    np.array(list(map(ord, decoded_tokens))) + self.min_token
                )
                decoded_dct_coeff = decoded_dct_coeff.reshape(-1, self.action_dim)
                assert decoded_dct_coeff.shape == (self.time_horizon, self.action_dim)
            except Exception as e:  # noqa: BLE001 - mirror upstream: never crash on a bad decode
                print(f"Error decoding tokens: {e}")
                decoded_dct_coeff = np.zeros((self.time_horizon, self.action_dim))
            decoded_actions.append(
                idct(decoded_dct_coeff / self.scale, axis=0, norm="ortho")
            )
        return np.stack(decoded_actions)


def load_fast_action_tokenizer(path: str | Path) -> FastActionTokenizer:
    """Rebuild a saved FAST tokenizer offline (no AutoProcessor / network).

    Expects a directory as saved by ``physical-intelligence/fast``:
      - ``bpe_tokenizer/tokenizer.json`` (the fit byte-level BPE)
      - ``processor_config.json`` (scale / vocab_size / min_token / action_dim / time_horizon)
    """
    from transformers import PreTrainedTokenizerFast

    path = Path(path)
    tok_file = path / "bpe_tokenizer" / "tokenizer.json"
    if not tok_file.exists():
        # The official ``physical-intelligence/fast`` repository stores these
        # files at its root.  Custom tokenizers saved by older code may use the
        # nested layout, so support both without changing the token math.
        tok_file = path / "tokenizer.json"
    if not tok_file.exists():
        raise FileNotFoundError(
            f"FAST tokenizer file not found under {path}. Expected the official "
            "physical-intelligence/fast layout (tokenizer.json + processor_config.json) "
            "or a legacy bpe_tokenizer/tokenizer.json layout."
        )
    bpe = PreTrainedTokenizerFast(
        tokenizer_file=str(tok_file), clean_up_tokenization_spaces=False
    )

    cfg = json.loads((path / "processor_config.json").read_text())
    return FastActionTokenizer(
        bpe,
        scale=float(cfg.get("scale", 10.0)),
        vocab_size=int(cfg.get("vocab_size", 1024)),
        min_token=int(cfg.get("min_token", 0)),
        action_dim=cfg.get("action_dim"),
        time_horizon=cfg.get("time_horizon"),
    )
