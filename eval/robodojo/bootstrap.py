"""Select an external RoboDojo/XPolicyLab checkout without copying model code."""

from __future__ import annotations
import sys
from pathlib import Path

REQUIRED = (
    "XPolicyLab/model_template.py",
    "XPolicyLab/client_server/ws/model_server.py",
    "XPolicyLab/utils/process_data.py",
    "XPolicyLab/utils/camera_extrinsics.py",
    "XPolicyLab/utils/pose_transform.py",
    "XPolicyLab/utils/setup_env_client.sh",
    "XPolicyLab/utils/run_sim_env_client.sh",
    "scripts/eval_policy.sh",
    "env_cfg/arx_x5.yml",
    "env_cfg/robot/_robot_info.json",
)


def bootstrap(root: Path) -> None:
    missing = [item for item in REQUIRED if not (root / item).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Incompatible RoboDojo checkout {root}: missing {missing}"
        )
    for path in (root / "XPolicyLab", root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
