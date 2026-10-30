"""Check user-provided weights and prepared teacher sources without downloading them."""

import argparse
from magic_w0.training.config import load_recipe
from magic_w0.training.resources import check_resources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--imports", action="store_true")
    args = parser.parse_args()
    check_resources(load_recipe(args.config), check_imports=args.imports)


if __name__ == "__main__":
    main()
