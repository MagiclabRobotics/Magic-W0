# Training MagicW0

[README](../README.md) · [Data](data.md) · [Checkpoints](checkpoints.md)

The training implementation is the MagicW0 architecture used by the released
policy: a Qwen3.5 backbone with Action, 2D World and 3D World experts. The 3D
expert predicts both geometry and motion.

## Example configurations

The files under `configs/train/` are minimal usage examples, not the original
experiment recipes. Learning rates, loss weights, optimizer and run length are
illustrative. Adjust them for your dataset and hardware. Structural dimensions,
action mappings and coordinate conventions must remain compatible with your
checkpoint and data. Additional model options use the configuration class defaults.
The validation results cover the implementation; they are not convergence
results for these example hyperparameters.

## Install

Use a clean Linux / Python 3.11 environment and NVIDIA CUDA GPUs. Install the
training runtime validated with two A800 80GB GPUs:

```bash
bash scripts/install.sh --train
```

The training installer uses PyTorch 2.7.1 / torchvision 0.22.1 with CUDA 12.6
wheels and constrains subsequent package resolution with `constraints/train.txt`.
It differs from the original inference installer (`--inference`, CUDA 12.1).
Use an NVIDIA driver compatible with the selected CUDA runtime.

The full-model functional check used per-GPU batch size 1, two full A800 80GB
GPUs, BF16, and gradient checkpointing. This is a tested configuration, not a
minimum VRAM guarantee; long histories and larger batches increase memory use.
Inference alone allocated about 8.32 GB in the checkpoint smoke test, excluding
the simulator.

W&B logging is optional (`pip install -e '.[logging]'`); recipes disable it by
default. DDP is the default distributed strategy. The framework also includes
PyTorch FSDP and ZeRO-2 strategies; the supplied recipes use DDP. The default kernels are SDPA and the
PyTorch DeltaNet implementation. Accelerated kernels require a separately
matched CUDA environment and are not enabled implicitly.

## Resources

Follow the [model download guide](models.md) for all resource URLs, pinned
revisions and expected destinations.

Download `pretrain_models/magic_w0_pretrain.pt` from
[XuhuaX/Magic-W0](https://huggingface.co/XuhuaX/Magic-W0) into `checkpoints/` with
the same relative path. For fresh pretraining, prepare Qwen3.5-2B base weights
and processor assets. Fine-tuning a complete policy checkpoint initializes the
VLM from that checkpoint, so its Qwen directory needs configuration, tokenizer
and processor files but not a second copy of base model tensors.

World-supervised training additionally needs:

| Resource | Recipe field |
| --- | --- |
| DINOv3 ViT-L/16 | `model.params.dinov3_checkpoint` |
| Track4World source | `model.params.track4world_project_root` |
| Track4World weights | `model.params.track4world_ckpt` |
| Depth Anything 3 backbone | `model.params.track4world_da3_backbone_path` |
| ConvNeXt-Tiny weights | `model.params.track4world_convnext_tiny_weights_path` |
| FAST tokenizer, when enabled | `model.params.fast_tokenizer_path` |

The four teacher/backbone weights are available with their original
configuration files and license notices:

```bash
hf download XuhuaX/Magic-W0 \
  --revision 5045b451991fd907cf602bae9a24e941ddfb55ba \
  --include 'third_party_models/**' \
  --local-dir "${MAGIC_W0_MODEL_ROOT:-checkpoints}"
```

Qwen and the optional FAST tokenizer are downloaded separately as described in
[Models](models.md). These
third-party models retain their respective licenses.

Prepare the pinned public Track4World source with the included training feature
adapter:

```bash
python scripts/prepare_track4world.py
python scripts/prepare_track4world.py --check
python scripts/check_training_resources.py --config configs/train/robodojo_sft.yaml --imports
```

See [third-party preparation](../third_party/README.md). The patch supplies
interfaces absent from the original teacher, so an unpatched checkout is not
sufficient. Source and weights retain their upstream terms. Do not blindly
install its complete requirements over this environment: its Transformers pin
can conflict with Qwen3.5. Large source archives and weights are not bundled.
The source preparation script never downloads weights; fetch the separately
licensed weights using the commands above. Weight paths are repository-relative or absolute; only
`track4world_config_path` is relative to the teacher source root.

The default local layout preserves the supplied teacher/backbone names:

```text
checkpoints/third_party_models/
  track4world_da3.pth
  convnext_tiny-983f1562.pth
  dinov3-vitl16-pretrain-lvd1689m/
  DA3NESTED-GIANT-LARGE-1.1/
```

The shared model settings live in `configs/train/model.yaml`. Model and data
roots can be relocated without editing every source:

```bash
export MAGIC_W0_MODEL_ROOT=/path/to/checkpoints
export MAGIC_W0_DATA_ROOT=/path/to/data
export MAGIC_W0_CACHE_ROOT=/path/to/cache
```

These roots default to `checkpoints`, `data` and `.cache/magic_w0`. Runtime paths
are relative to the repository root. YAML `include` paths are relative to the
including YAML. Child values override parent values; lists replace whole lists.

## RoboDojo fine-tuning

Follow [data preparation](data.md), then inspect the resolved recipe:

```bash
bash scripts/train.sh --config configs/train/robodojo_sft.yaml --dry-run
python scripts/check_data.py --config configs/train/robodojo_sft.yaml --samples 2
NPROC_PER_NODE=8 bash scripts/train.sh --config configs/train/robodojo_sft.yaml
```

The fine-tuning recipe initializes from the published pretraining checkpoint.
It uses full model fine-tuning, teacher-supervised world losses, and the released
25-frame / interval-20 history schedule. Batch size is deliberately conservative;
adjust `data.batch_size`, worker count and optimizer settings for your GPUs.
The recipe is a portable starting point, not a claim that a different dataset
will reproduce the published benchmark score.

Each run saves its resolved YAML and software versions in the output directory,
even when W&B is disabled. Resumed invocations keep the earlier metadata and
write a new timestamped copy.

To initialize from another compatible policy checkpoint:

```bash
bash scripts/train.sh --config configs/train/robodojo_sft.yaml \
  --init-checkpoint checkpoints/robodojo_sft_models/magic_w0_robodojo.pt
```

For a short verification run, add `--max-steps 2 --output-dir outputs/smoke`.
This still runs the actual model and requires its resources and data.

## Multi-source pretraining

Edit the independently mapped and normalized sources in
`configs/train/data/pretrain.yaml`. The two sources there demonstrate the
canonical 34D contract; they are not bundled datasets or the original corpus.
Pretraining starts from Qwen base weights and newly initialized experts:

```bash
python scripts/check_data.py --config configs/train/pretrain.yaml --list
bash scripts/train.sh --config configs/train/pretrain.yaml --dry-run
NPROC_PER_NODE=8 bash scripts/train.sh --config configs/train/pretrain.yaml
```

The data implementation also supports batch-level VLM JSONL co-training.
Enable a `vlm_jsonl` source and its `cotrain_vlm_modalities` only after preparing
the corresponding chat/image data. Source weights control sampling; keep each
source's action mapping, masks and statistics explicit.

## Resume and multi-node training

Resume restores complete training state; initialization only restores model
weights. A missing resume checkpoint is an error:

```bash
bash scripts/train.sh --config configs/train/robodojo_sft.yaml \
  --resume outputs/robodojo_sft
```

A directory selects the highest numeric checkpoint step, not the newest mtime.
Do not change the data layout or mixture while resuming an exact training run.
Resume also checks model training settings, including history, loss weights and
stream visibility. Relocated resource paths and execution backends are allowed;
the assets at those paths must remain the same. Use `--init-checkpoint` to change
the training objective or begin a new stage with a fresh optimizer.

On each node, set the same master address, port and node count; give each node a
different zero-based `NODE_RANK`:

```bash
NNODES=2 NODE_RANK=0 MASTER_ADDR=<master-ip> MASTER_PORT=29500 \
NPROC_PER_NODE=8 bash scripts/train.sh --config configs/train/pretrain.yaml
```

Use `NODE_RANK=1` on the other node. Data, model assets and the run directory
must be accessible consistently on all nodes.

Finally [export a policy checkpoint](checkpoints.md) and use the existing
[RoboDojo inference guide](robodojo.md) to evaluate it.
