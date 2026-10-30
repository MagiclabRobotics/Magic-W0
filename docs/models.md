# Models and resources

[Home](../README.md) · [Training](training.md) · [Data](data.md)

The policy and teacher files below can be downloaded without authentication.
Third-party weights retain their own licenses; the repository's MIT license
covers neither all dependencies nor all model weights. See their model cards
and [NOTICE](../NOTICE). Track4World’s license restricts it to academic use;
DA3NESTED-GIANT-LARGE-1.1 weights use CC-BY-NC-4.0. These restrictions apply
to the world-teacher training recipe even though the MagicW0 repository has
an MIT license. DINOv3 weights retain the DINOv3 license supplied with them.

Commands run from the repository root. `hf` is installed with the package's
Transformers / Hugging Face dependencies.

## Inference policy

```bash
python scripts/prepare_resources.py --download-qwen \
  --checkpoint-repo XuhuaX/Magic-W0 \
  --checkpoint-file robodojo_sft_models/magic_w0_robodojo.pt \
  --checkpoint-revision 78e6afa2f07550c602bd6538fb99cb8a0b8f913f
python scripts/prepare_resources.py --check
```

This prepares the policy at `checkpoints/magic_w0_robodojo.pt` and pinned Qwen
processor assets at `checkpoints/qwen3.5-2b-assets/`. Base Qwen tensors are not
needed for this policy. The checkpoint is 7,228,809,258 bytes; its SHA-256 is
`cf21ef5a43f929dc68eacc8099bf4366cba1146c337a802f3c43723c74e5ff25`.

## Fine-tuning initialization

Download the pretraining policy to the path expected by the SFT recipe:

```bash
hf download XuhuaX/Magic-W0 pretrain_models/magic_w0_pretrain.pt \
  --revision 5045b451991fd907cf602bae9a24e941ddfb55ba \
  --local-dir checkpoints
python scripts/prepare_resources.py --download-qwen
```

## Fresh pretraining

Fresh pretraining needs Qwen base weights as well as processor files:

```bash
hf download Qwen/Qwen3.5-2B \
  --revision 15852e8c16360a2fea060d615a32b45270f8a8fc \
  --local-dir checkpoints/Qwen3.5-2B
hf download physical-intelligence/fast \
  --revision ec4d7aa71691cac0b8bed6942be45684db2110f4 \
  --include tokenizer.json processor_config.json README.md \
  --local-dir checkpoints/fast_tokenizer
```

FAST is only required when `use_fast_action_tokens: true`; the SFT example
leaves it disabled. The loader uses local tokenizer assets without executing
Hugging Face remote Python code.

## World teachers

Both example recipes use world supervision. Download the DINOv3, Track4World,
Depth Anything 3 and ConvNeXt resources, retaining their configuration and
license files:

```bash
hf download XuhuaX/Magic-W0 \
  --revision 5045b451991fd907cf602bae9a24e941ddfb55ba \
  --include 'third_party_models/**' \
  --local-dir checkpoints
python scripts/prepare_track4world.py
```

Expected layout:

```text
checkpoints/
  qwen3.5-2b-assets/                    # Inference / SFT processor assets
  Qwen3.5-2B/                          # Fresh pretraining only
  fast_tokenizer/                      # FAST pretraining only
  magic_w0_robodojo.pt                 # Inference policy
  pretrain_models/magic_w0_pretrain.pt  # SFT initialization
  third_party_models/
    track4world_da3.pth
    convnext_tiny-983f1562.pth
    dinov3-vitl16-pretrain-lvd1689m/
    DA3NESTED-GIANT-LARGE-1.1/
```

Prepare source mappings and normalization using the [data guide](data.md), then
check resources before GPU allocation:

```bash
python scripts/check_training_resources.py \
  --config configs/train/robodojo_sft.yaml --imports
```

For relocated model paths, set `MAGIC_W0_MODEL_ROOT` and use the same destination
in the download commands. Files under `checkpoints/` are excluded from Git.
