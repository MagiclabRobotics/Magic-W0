"""Validate selected training data using the same reader and processor as training."""

import argparse
import json

from magic_w0.data.config import DatasetGroupConfig, source_config_from_dict
from magic_w0.data.backends import get_data_backend
from magic_w0.models import build_data_processor
from magic_w0.training.config import load_recipe


def leaves(source):
    if isinstance(source, DatasetGroupConfig):
        for child in source.datasets:
            yield from leaves(child)
    elif source is not None:
        yield source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-index", type=int)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    config = load_recipe(args.config)
    sources = [
        leaf
        for raw in config["data"]["datasets"]
        for leaf in leaves(source_config_from_dict(raw))
    ]
    if args.source_index is not None and not 0 <= args.source_index < len(sources):
        parser.error("--source-index is outside the enabled source list")
    for index, source in enumerate(sources):
        if args.source_index is not None and index != args.source_index:
            continue
        print(json.dumps({"index": index, "type": source.type, "root": source.root}))
        if args.list:
            continue
        source.lerobot_version = source.lerobot_version or config["data"].get(
            "lerobot_version", "v2.1"
        )
        dataset = get_data_backend(source.type).make_dataset(source)
        processor = build_data_processor(
            config["model"]["name"],
            config["model"].get("processor"),
            model_params=config["model"].get("params", {}),
        )
        dataset = processor.transform_dataset(dataset, source)
        if len(dataset) < 1:
            raise ValueError(f"Source {index} is empty")
        for sample in range(min(args.samples, len(dataset))):
            item = dataset[sample]
            print(
                json.dumps(
                    {
                        "sample": sample,
                        "fields": {
                            key: list(value.shape)
                            if hasattr(value, "shape")
                            else type(value).__name__
                            for key, value in item.items()
                        },
                    }
                )
            )


if __name__ == "__main__":
    main()
