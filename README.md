<div align="center">

<img src="assets/magic-w0-color-logo-v2.png" alt="Magic-W0" width="520">

# Magic-W0: A Structured World–Action Foundation Model for Physical Intelligence

*Magic-Lab Team · Magiclab Robotics Inc.*

<a href="https://github.com/MagiclabRobotics/Magic-W0"><img src="https://img.shields.io/badge/Repository-GitHub-black?logo=github" alt="GitHub Repository"></a> <img src="https://img.shields.io/badge/Code_%26_Weights-Coming_Soon-lightgrey" alt="Code and weights coming soon">

[Update News](#update-news) · [Abstract](#abstract) · [Key Features](#key-features) · [Getting Started](#getting-started)

</div>

## Update News

- **[2026-09-30]** Added the project overview. Code and pretrained checkpoints are coming soon.

## Abstract

Magic-W0 learns structured world transitions and continuous robot actions from diverse embodied experience. It represents the current scene, action-induced 3D motion, and task-relevant future semantics, and couples world prediction with action generation through layer-aligned interaction. Pretrained on approximately 2.014M effective action episodes and 2.61M visual-language samples, it achieves 99.05% average success on LIBERO and 94.6% across five real-robot tasks after downstream fine-tuning.

<p align="center">
  <img src="assets/figure1-v9.gif" alt="Magic-W0 overview: diverse embodied data, a unified action interface, structured world modeling, and robot manipulation" width="100%">
</p>

## Key Features

- **World–action modeling:** learn geometry, motion, future semantics, and continuous control together.
- **Cross-embodiment data:** use a shared 34D state–action interface with masks for missing dimensions.
- **Teacher-free policy execution:** use Track4World and DINOv3 supervision during training without running these teachers at inference.
- **Configurable training workflows:** adapt YAML recipes for pretraining, fine-tuning, and distributed execution.
- **Checkpoint diagnostics:** check inference contracts and evaluate generated actions and world representations.

## Getting Started

### Installation

Coming soon: environment requirements, dependency installation, and a setup check.

### Model Checkpoints

Coming soon: checkpoint downloads, required assets, and model loading examples.

### Data Preparation

Coming soon: supported data formats, custom dataset configuration, and normalization instructions.

### Training and Fine-tuning

Coming soon: training configurations, single-node and multi-node launch commands, fine-tuning, and checkpoint resumption.

### Inference and Deployment

Coming soon: a minimal inference example, observation preparation, action decoding, and robot integration instructions.

### Evaluation

Coming soon: checkpoint diagnostics, benchmark setup, and evaluation commands.
