"""CLI bridge for the distributed trainer package."""

from __future__ import annotations


def main() -> None:
    """Run the distributed trainer CLI."""
    from magic_w0.training.cli import train_main

    train_main()


if __name__ == "__main__":
    main()
