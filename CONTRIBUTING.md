# Contributing

Please open an [issue](https://github.com/MagiclabRobotics/Magic-W0/issues) for
bugs or questions, or submit a pull request with a focused change.

Use Python 3.11 and the training environment for development:

```bash
bash scripts/install.sh --train
python -m pip install -c constraints/train.txt -e '.[test]'
python -m ruff check src eval scripts tests
python -m pytest -q
```

The CPU suite uses small models and synthetic fixtures. Tests requiring local
Qwen processor assets or real datasets skip when those resources are absent.
GPU training and RoboDojo / Isaac Sim integration are separate checks; follow
[Training](docs/training.md) and [RoboDojo](docs/robodojo.md).

For a bug report, include the command, relevant configuration, Python / PyTorch /
Transformers versions, GPU model, and the first error traceback. Changes to
state/action mappings should include the units, reference frames and masks.
Third-party code must retain its original license notices.
