<div align="center">

<img src="assets/magic-w0-color-logo-v2.png" alt="Magic-W0" width="520">

# Magic-W0: A Structured World–Action Foundation Model for Physical Intelligence

*Magic-Lab Team · Magiclab Robotics Inc.*

<a href="https://embodied.magiclab.top/works/wam/magic-w0/index.html"><img src="https://img.shields.io/badge/Website-Project_Page-blue" alt="Project Homepage"></a> <a href="https://github.com/MagiclabRobotics/Magic-W0"><img src="https://img.shields.io/badge/Repository-GitHub-black?logo=github" alt="GitHub Repository"></a> <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License"></a>
<br>
<a href="https://huggingface.co/Flyfish101/Magic-W0"><img src="https://img.shields.io/badge/%F0%9F%A4%97_Model-HuggingFace-yellow" alt="Model on HuggingFace"></a>

[Update News](#update-news) · [Abstract](#abstract) · [Key Features](#key-features) · [Getting Started](#getting-started)

</div>

<a name="update-news"></a>

## 📣 Update News

- **[2026-10-05]** Added the RoboDojo inference adapter and deployment configuration.
- **[2026-09-30]** Added the project overview.

<a name="abstract"></a>

## 📖 Abstract

Magic-W0 learns structured world transitions and continuous robot actions from diverse embodied experience. It represents the current scene, action-induced 3D motion, and task-relevant future semantics, and couples world prediction with action generation through layer-aligned interaction. Pretrained on approximately 2.014M effective action episodes and 2.61M visual-language samples, it achieves 99.05% average success on LIBERO and 94.6% across five real-robot tasks after downstream fine-tuning.

<p align="center">
  <img src="assets/figure1-v9.gif" alt="Magic-W0 overview: diverse embodied data, a unified action interface, structured world modeling, and robot manipulation" width="100%">
</p>

<a name="key-features"></a>

## ✨ Key Features

- 🌍 **World–action modeling:** learn geometry, motion, future semantics, and continuous control together.
- 🤖 **Cross-embodiment data:** use a shared 34D state–action interface with masks for missing dimensions.
- ⚡ **Teacher-free policy execution:** use Track4World and DINOv3 supervision during training without running these teachers at inference.
- 🕒 **Visual history:** retain checkpoint-configured past observations independently for each environment.
- 🛠️ **RoboDojo integration:** serve batched actions with checkpoint validation, normalization, and camera-frame EEF conversion.

<a name="getting-started"></a>

## 🚀 Getting Started

This repository provides the Magic-W0 inference runtime and RoboDojo adapter.
Training and fine-tuning recipes will be released separately.

### Installation

Use Linux, Python 3.11, and an NVIDIA GPU. Run from the repository root:

```bash
conda create -n magic-w0 python=3.11 -y
conda activate magic-w0
bash scripts/install.sh
```

The installer uses PyTorch 2.5.1 / CUDA 12.1. Install RoboDojo, Isaac Sim, assets,
and the required XPolicyLab extensions in a separate simulator environment;
see the [inference guide](docs/robodojo.md).

### Model Resources

Model repository: [🤗 Hugging Face](https://huggingface.co/Flyfish101/Magic-W0).
Prepare a compatible local policy checkpoint and the pinned Qwen configuration,
tokenizer, and processor assets:

```bash
python scripts/prepare_resources.py \
  --checkpoint-source /path/to/magic_w0_robodojo.pt \
  --download-qwen
python scripts/prepare_resources.py --check
```

Files are prepared under `checkpoints/`. Qwen base model weights are unnecessary;
the policy checkpoint contains the learned VLM tensors.

### RoboDojo Evaluation

```bash
export ROBODOJO_ROOT=/path/to/RoboDojo
bash eval/robodojo/run.sh \
  --task stack_bowls --seed 0 --episodes 10 \
  --policy-name Magic_W0_Local \
  --policy-gpu 0 --env-gpu 1 --sim-env RoboDojo
```

For a single GPU, use `--env-gpu 0` and reduce the simulator's environment count.
Select the simulator configuration with `--env-config YOUR_ARX_ENV_CONFIG`.
Add `--dry-run` to inspect the launch commands. Deployment defaults are in
[`configs/robodojo.yaml`](configs/robodojo.yaml): **10 inference steps** and
**20 executed actions per replan**. Logs and resolved settings are saved under
`runs/robodojo/`; task success is recorded in RoboDojo's native result files.
See the [inference guide](docs/robodojo.md) for resource overrides and multiple tasks.

## 🙏 Acknowledgements

We thank the Hugging Face and LeRobot communities for their infrastructure and tooling. Magic-W0 builds on Qwen3.5, Track4World, Depth Anything 3, DINOv3, and FAST. We also thank the LIBERO and RoboDojo teams for their evaluation benchmarks.

## 📚 Citation

If you find Magic-W0 useful for your research, please cite our paper:

```bibtex
@article{chen2026magicw0,
  title   = {Magic-W0: A Structured World--Action Foundation Model for Physical Intelligence},
  author  = {Chen, Xuhua and Yin, Zhenhan and Zhang, Yuan and Zhang, Tao and others},
  journal = {arXiv preprint arXiv:2609.39870},
  year    = {2026},
  url     = {https://arxiv.org/abs/2609.39870}
}
```

## 📜 License

Released under the **MIT License**. See [LICENSE](LICENSE) for the full terms.
Third-party components retain their own licenses; see [NOTICE](NOTICE).
