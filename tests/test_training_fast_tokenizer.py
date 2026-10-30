"""The packaged FAST path must load real local BPE assets without Pi05 code."""

import json

import numpy as np
import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from magic_w0.models.action_backbone import modeling_magicw0_base as modeling


def save_fast_assets(root, *, nested=False):
    root.mkdir(parents=True)
    bpe = Tokenizer(models.BPE(unk_token="[UNK]"))
    bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    bpe.decoder = decoders.ByteLevel()
    bpe.train_from_iterator(
        ["@" * 200, "".join(chr(i) for i in range(128))],
        trainer=trainers.BpeTrainer(
            vocab_size=512,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=["[UNK]"],
            show_progress=False,
        ),
    )
    directory = root / "bpe_tokenizer" if nested else root
    directory.mkdir(exist_ok=True)
    bpe.save(str(directory / "tokenizer.json"))
    (root / "processor_config.json").write_text(
        json.dumps(
            {"scale": 10.0, "min_token": -64, "vocab_size": bpe.get_vocab_size()}
        )
    )


@pytest.mark.parametrize("nested", [False, True])
def test_packaged_fast_loader_encodes_and_decodes_local_assets(tmp_path, nested):
    assets = tmp_path / "fast"
    save_fast_assets(assets, nested=nested)
    fast = modeling._load_fast_action_tokenizer(str(assets))
    actions = np.random.default_rng(7).uniform(-0.5, 0.5, (2, 8, 3)).astype(np.float32)
    tokens = fast(actions)
    assert len(tokens) == 2 and all(ids for ids in tokens)
    assert all(0 <= token < fast.vocab_size for ids in tokens for token in ids)
    decoded = fast.decode(tokens)
    assert decoded.shape == actions.shape
    # Orthonormal DCT preserves the RMS bound of coefficient quantization.
    assert np.sqrt(np.mean((decoded - actions) ** 2)) <= 0.5 / fast.scale + 1e-6


def test_fast_relative_checkout_path_resolves_outside_working_directory(
    tmp_path, monkeypatch
):
    checkout = tmp_path / "checkout"
    save_fast_assets(checkout / "checkpoints/fast")
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.chdir(outside)
    monkeypatch.setattr(
        modeling,
        "__file__",
        str(checkout / "src/magic_w0/models/action_backbone/modeling_magicw0_base.py"),
    )
    fast = modeling._load_fast_action_tokenizer("checkpoints/fast")
    assert fast(np.zeros((1, 8, 3), dtype=np.float32))[0]
