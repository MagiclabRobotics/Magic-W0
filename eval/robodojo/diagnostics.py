"""RoboDojo inference adapter components."""

from __future__ import annotations
import csv
import json
from typing import Any
import numpy as np
from .layout import (
    _CAMERA_ALIASES,
    camera_eef_labels,
    camera_eef_layout_spec,
    joint_eef_union32_labels,
)


from .paths import REPO_ROOT

_POLICY_DIR = REPO_ROOT


class DiagnosticsMixin:
    def _append_trajectory_rows(self, kind, env_indices, rows) -> None:
        trajectory_dir = self.model_input_dump_dir / "trajectories"
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        for env_idx, row in zip(env_indices, rows, strict=True):
            values = row["state"] if kind == "measured_state" else row
            values = np.asarray(values, dtype=np.float32).reshape(-1)
            if values.shape[0] == 32 and self.model_input_layout == "joint_eef_union32":
                labels = joint_eef_union32_labels()
            elif values.shape[0] == 14:
                labels = joint_eef_union32_labels()[:14]
            elif (
                values.shape[0] == camera_eef_layout_spec(self.camera_eef_layout)["dim"]
            ):
                labels = camera_eef_labels(self.camera_eef_layout)
            else:
                raise ValueError(f"unsupported trajectory row shape {values.shape}")
            path = trajectory_dir / f"env{env_idx}_{kind}.csv"
            token = (kind, env_idx)
            needs_header = token not in self._trajectory_files_initialized
            sample = self._trajectory_sample_counts.get(token, 0)
            with path.open("a", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                if needs_header:
                    writer.writerow(["sample"] + labels)
                    self._trajectory_files_initialized.add(token)
                writer.writerow([sample] + values.tolist())
            self._trajectory_sample_counts[token] = sample + 1

    def _dump_model_inputs(
        self,
        env_indices,
        observations,
        padded_states,
        normalized_states,
        state_dim_mask,
        batch,
        prepared,
    ) -> None:
        """Save exact pre-model images and first-step state diagnostics."""
        from PIL import Image, ImageDraw

        vlm_inputs = prepared.get("vlm_inputs", {})
        tensor_metadata = {}
        for key, value in vlm_inputs.items():
            if not hasattr(value, "shape"):
                continue
            entry: dict[str, Any] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            if getattr(value, "is_floating_point", lambda: False)() and value.numel():
                numeric = value.detach().float()
                entry.update(
                    min=float(numeric.min().cpu()),
                    max=float(numeric.max().cpu()),
                    mean=float(numeric.mean().cpu()),
                )
            elif key == "image_grid_thw":
                entry["values"] = value.detach().cpu().tolist()
            tensor_metadata[key] = entry

        for batch_index, env_idx in enumerate(env_indices):
            replan_step = self._replan_steps.get(env_idx, 0)
            if replan_step != 0 and not self.save_model_input_sequence:
                continue
            target = self.model_input_dump_dir / f"env{env_idx}_replan{replan_step:03d}"
            target.mkdir(parents=True, exist_ok=True)
            camera_images = []
            camera_names = []
            for key, short_name in zip(
                self.policy.config.camera_keys, _CAMERA_ALIASES, strict=True
            ):
                chw = batch[key][batch_index].detach().cpu().numpy()
                rgb = np.transpose(chw, (1, 2, 0)).astype(np.uint8, copy=False)
                image = Image.fromarray(rgb, mode="RGB")
                image.save(target / f"{short_name}_model_input.png")
                camera_images.append(image)
                camera_names.append(short_name)

            # A full sequence only needs the exact images.  Robot states are
            # already recorded in trajectories/envN_measured_state.csv.
            if replan_step != 0:
                print(f"[Magic_W0][inputs] saved {target}", flush=True)
                continue

            montage = Image.new("RGB", (256 * len(camera_images), 286), "white")
            draw = ImageDraw.Draw(montage)
            for index, (name, image) in enumerate(zip(camera_names, camera_images)):
                montage.paste(image, (index * 256, 30))
                draw.text((index * 256 + 6, 8), name, fill="black")
            montage.save(target / "camera_inputs_montage.png")

            raw = np.asarray(observations[batch_index]["state"], dtype=np.float32)
            padded = padded_states[batch_index]
            normalized = normalized_states[batch_index]
            mask = state_dim_mask[batch_index]
            np.savez_compressed(
                target / "robot_state_model_input.npz",
                raw_robot_state=raw,
                padded_state_32d=padded,
                normalized_state_32d=normalized,
                state_dim_mask_32d=mask,
                normalization_low_32d=self.stats.state_low,
                normalization_high_32d=self.stats.state_high,
            )
            payload = {
                "env_idx": env_idx,
                "layout_id": observations[batch_index]["layout_id"],
                "task": observations[batch_index]["task"],
                "image_preprocess": {
                    "source_layout": "RGB uint8 CHW",
                    "output_shape": [
                        3,
                        self.policy.config.image_size,
                        self.policy.config.image_size,
                    ],
                    "resize": "aspect-preserving bilinear",
                    "padding": "center zero padding",
                },
                "raw_robot_state": raw.tolist(),
                "padded_state_32d": padded.tolist(),
                "normalized_state_32d": normalized.tolist(),
                "state_dim_mask_32d": mask.tolist(),
                "normalization_low_32d": self.stats.state_low.tolist(),
                "normalization_high_32d": self.stats.state_high.tolist(),
                "vlm_inputs": tensor_metadata,
            }
            (target / "model_input.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )

            width, height = 1280, 600
            chart = Image.new("RGB", (width, height), "white")
            chart_draw = ImageDraw.Draw(chart)
            chart_draw.text(
                (20, 12), f"env={env_idx} normalized 32-D robot state", fill="black"
            )
            zero_y, scale = 280, 240
            chart_draw.line(
                (20, zero_y, width - 20, zero_y), fill=(80, 80, 80), width=2
            )
            bar_width = (width - 40) / self.state_dim
            for dim, value in enumerate(normalized):
                x0 = int(20 + dim * bar_width + 2)
                x1 = int(20 + (dim + 1) * bar_width - 2)
                y = int(zero_y - float(value) * scale)
                color = (45, 110, 210) if mask[dim] else (190, 190, 190)
                chart_draw.rectangle(
                    (x0, min(y, zero_y), x1, max(y, zero_y)), fill=color
                )
                chart_draw.text((x0, 535), str(dim), fill="black")
            chart_draw.text(
                (20, 565),
                "blue=valid RoboDojo dimension; gray=padded dimension",
                fill="black",
            )
            chart.save(target / "robot_state_visualization.png")
            print(f"[Magic_W0][inputs] saved {target}", flush=True)
