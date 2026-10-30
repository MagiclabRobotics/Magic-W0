# Checkpoints

| Format | Contents | Purpose |
| --- | --- | --- |
| Training checkpoint | Model, config, normalization, training state and optimizer/scheduler state | Continue the same run |
| Inference checkpoint | Model tensors, inference config and normalization | Existing MagicW0 policy runtime |

Fine-tuning can initialize model weights from either a complete compatible
training checkpoint or a published inference checkpoint. A published inference
checkpoint cannot restore optimizer state or the original data cursor.

## Export

```bash
python scripts/export_checkpoint.py \
  --input outputs/robodojo_sft/checkpoint-epoch-0-step-2000.pt \
  --output checkpoints/magic_w0_custom.pt
```

The input can also be a run directory. Export refuses to overwrite an existing
file. It retains VLM, Action, 2D World, 3D World and state projection weights,
along with normalization metadata. Optimizer/scheduler state and teacher
weights are omitted. Attention backend fields are set to the public runtime's
SDPA / torch implementation without altering model tensors.

New training exports record the quantile scale epsilon used by the data
processor. Deployment then uses the same scale for constant dimensions without
rewriting the saved normalization bounds. Existing released checkpoints retain
their original normalization behavior.

The exporter checks the supported history/visibility graph and rejects a
four-expert checkpoint, missing model branches or absent normalization. Use
the policy checkpoint override in the [RoboDojo guide](robodojo.md) to load the
result. Input camera conventions and selected normalization source must match
the training data.

Do not edit parameter names or use partial loading to force a structurally
different checkpoint into this runtime. The public implementation targets the
MagicW0 three-stream architecture; normalization and action conventions remain
part of the deployment contract.

The supplied inference exporter targets history-enabled fine-tuned policies.
An unfinished pretraining run with history disabled remains a training
checkpoint for later fine-tuning; it is not silently converted to a different
inference history recipe.
