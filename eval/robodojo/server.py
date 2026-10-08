"""Run the existing XPolicyLab WebSocket transport with the local policy."""

import argparse
import asyncio
import json
import os
from pathlib import Path
import yaml
from .bootstrap import bootstrap
from .paths import resolve_robodojo_root


async def serve(server, config):
    ready = Path(config["ready_file"]) if config.get("ready_file") else None
    try:
        await server.start()
        if ready:
            record = {"run_id": config["run_id"], "pid": os.getpid(), "url": server.url}
            temporary = ready.with_suffix(".tmp")
            temporary.write_text(json.dumps(record))
            temporary.replace(ready)
        await server.serve_forever()
    finally:
        if ready:
            ready.unlink(missing_ok=True)
        await server.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    bootstrap(resolve_robodojo_root(config))
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig
    from .policy import Model

    model = Model(config)
    server = PolicyServer(
        model,
        PolicyServerConfig(
            host=config["host"],
            port=int(config["port"]),
        ),
    )
    asyncio.run(serve(server, config))


if __name__ == "__main__":
    main()
