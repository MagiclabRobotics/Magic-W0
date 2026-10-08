# RoboDojo inference

Use a Linux machine with an NVIDIA GPU. Keep the model environment separate
from your existing RoboDojo / Isaac Sim environment. The simulator and its
assets are not included in this repository.

Run commands from the Magic-W0 repository root.

## Installation

```bash
conda create -n magic-w0 python=3.11 -y
conda activate magic-w0
bash scripts/install.sh
```

The package installs from `src/magic_w0`. The installer retains PyTorch 2.5.1
and CUDA 12.1 from the original inference export. It installs model and
WebSocket dependencies, rather than recreating the exported simulator environment.
Install RoboDojo and its assets separately following your benchmark setup.

## Resources

```bash
python scripts/prepare_resources.py \
  --qwen-source /path/to/Qwen3.5-2B \
  --checkpoint-source /path/to/magic_w0_robodojo.pt
python scripts/prepare_resources.py --check
```

Alternatively use `--download-qwen` to download only the pinned configuration,
tokenizer and processor files in `configs/resources.json`. Base Qwen model
weights are unnecessary: all learned VLM tensors come from the policy checkpoint.
Resources are stored in `checkpoints/` and excluded from Git.

The [public model repository](https://huggingface.co/Flyfish101/Magic-W0)
has no `.pt` file at the revision recorded in the manifest. Use an existing
trusted checkpoint. For another Hugging Face repository, use
`--checkpoint-repo`, `--checkpoint-file` and `--checkpoint-revision` with its
immutable commit SHA. Authentication uses the normal Hugging Face CLI cache;
never put a token in a config file. `HF_ENDPOINT` can select an approved mirror.

Preparation writes a resource receipt. `--check` rejects missing files and LFS
pointers; the server then validates architecture, normalization metadata,
history schedule and every model tensor before serving actions.

## Benchmark integration

```bash
export ROBODOJO_ROOT=/path/to/RoboDojo
bash eval/robodojo/run.sh \
  --task stack_bowls --seed 0 \
  --policy-gpu 0 --env-gpu 1 \
  --sim-env RoboDojo --dry-run
```

Remove `--dry-run` to run evaluation. The model uses the current Python;
`--policy-env magic-w0` selects another Conda environment. `--sim-env` also
accepts a Conda prefix. `--checkpoint` and `--vlm-assets` override resource paths.
Use `--policy-name Magic_W0_Local` to install a separate shim alongside an
existing adapter. `--env-config` selects an external benchmark configuration.
`--episodes 1` caps the integration run through RoboDojo's `EVAL_NUM` support.
For a small GPU, use a separate external simulator config with fewer environments;
retain `config_name: arx_x5` when reusing the existing ARX evaluation layouts.
For two-environment evaluation, copy `env_cfg/arx_x5.yml` to a
new environment config, retain its robot, scene, camera and observation settings,
and set `config.sim` to a dedicated YAML name under `env_cfg/sim/`. Copy the
original simulator settings there, changing only `scene.num_envs` to `2`.
Pass the new environment config name (without `.yml`) to `--env-config`.
Config paths are resolved relative to this repository, independent of the
simulator's working directory.

The launcher rejects occupied ports and waits for a run-specific readiness
record plus a WebSocket ping before starting the simulator. On exit it stops
the whole simulator/server process groups, including subprocesses that outlive
their launcher.

The external checkout must contain `scripts/eval_policy.sh`, `env_cfg/`, and
`XPolicyLab/` with the WebSocket transport, camera extrinsics and pose-transform
helpers listed in `eval/robodojo/bootstrap.py`. A plain RoboDojo checkout without
these XPolicyLab extensions is insufficient.

At launch a small `XPolicyLab/policy/Magic_W0` shim imports this repository's
deployment loop. No model code is copied. An existing unrelated policy at that
path is never overwritten; move it aside first. Launches sharing the same
RoboDojo checkout should run sequentially, because the shim's `deploy.yml` is shared.

`configs/robodojo.yaml` preserves camera-frame EEF control, `arx_x5`, SDPA,
10 inference steps and 20 executed actions per replan. History dimensions and
interval come from the checkpoint. Camera extrinsics are frozen for each action
chunk, and histories remain independent across active environments.

## Multiple tasks and seeds

```bash
python -m eval.robodojo.evaluate \
  --tasks stack_bowls YOUR_OTHER_TASK \
  --seeds 0 1 2 --policy-gpu 0 --env-gpu 1 --sim-env RoboDojo
```

Provide the task list for your benchmark version explicitly. Each run records
its resolved config, policy/simulator logs and process status under
`runs/robodojo/`. The batch summary records task/seed exit codes. Completion
means the simulator exited successfully; it does **not** imply task success.
Use RoboDojo's own result files for success rates. Automatic resume and metric
aggregation are not enabled until the external benchmark's result format is fixed.

## Code map and validation

| Path | Responsibility |
| --- | --- |
| `src/magic_w0/` | Model configuration, expert modules, processing and checkpoint tensor split |
| `eval/robodojo/policy.py` | Batched inference, per-environment history and replanning |
| `eval/robodojo/loading.py` | Strict checkpoint loading and normalization contracts |
| `eval/robodojo/observation.py`, `action.py` | Observation packing and reconstructed action targets |
| `eval/robodojo/robodojo_adapter.py`, `deploy.py` | Simulator boundary and episode loops |
| `configs/` | Deployment defaults and resource versions |
| `scripts/` | Installation and resource preparation |

```bash
python -m pip install -e '.[test]'
pytest
ruff check src eval scripts tests
```

CPU tests cover layouts, rotations, normalization, histories and the launch
contract. GPU inference and simulator integration require the real checkpoint,
assets and compatible RoboDojo checkout.

Before the simulator test, a real checkpoint can be tested on CUDA using a
synthetic observation:

```bash
python -m eval.robodojo.gpu_smoke \
  --robodojo-root "$ROBODOJO_ROOT" \
  --checkpoint /path/to/magic_w0_robodojo.pt \
  --vlm-assets /path/to/qwen3.5-2b-assets
```

This checks strict model loading and finite action output, and records inference
latency and peak model memory. It does not replace the real simulator closed loop.

For a longer CUDA regression, replace `gpu_smoke` with `stress_gpu` and add
`--output runs/gpu-stress.json`. The default three seeds exercise 28 replans
each, every three-camera ordering, a full visual-history window, shrinking
active batches and reset replay. Observations are synthetic; weights and
inference run on the real GPU.
