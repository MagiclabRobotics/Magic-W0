# Training data

Training reads LeRobot v2.1 or v3 data without importing the LeRobot training
framework. Start from one source whose format you have checked.

For v2.1, a typical dataset contains:

```text
data/robodojo/
  meta/info.json
  meta/tasks.jsonl
  meta/episodes.jsonl
  meta/episodes_stats.jsonl
  data/chunk-000/episode_000000.parquet
  videos/chunk-000/<camera-key>/episode_000000.mp4
```

The reader follows `data_path`, `video_path`, feature definitions and FPS in
`meta/info.json`. Videos must align with the episode's frame timestamps. Source
camera names must match the configured keys. V3 uses its own reader and metadata;
select the correct `lerobot_version` rather than renaming v3 files into v2 paths.

## Getting data

This code release does not bundle datasets or the original pretraining mixture.
Obtain RoboDojo demonstrations through the [official benchmark data documentation](https://robodojo-benchmark.com/doc/)
and prepare a LeRobot export matching the contract below. A raw download is not
a compatible training source until its fields, frames and masks match the YAML.
The example `data/robot_a` and `data/robot_b` paths are templates for your own
canonical 34D datasets.

## State/action contract

The published policy uses 34D state and action vectors:

| Slots | Meaning |
| --- | --- |
| 0–6 / 8–14 | Left / right joint slots |
| 7 / 15 | Left / right gripper |
| 16–18 / 25–27 | Left / right end-effector position |
| 19–24 / 28–33 | Left / right end-effector rotation in 6D representation |

Missing slots are zero-filled and masked. The RoboDojo training example maps a
32D joint/EEF union export into these slots and supervises grippers and EEF
channels. It expects the source masks `state_mask` and `action_mask`. In the 32D source,
left joints occupy 0–5, left gripper 6, right joints 7–12, right gripper 13,
left EEF position/rot6d 14–22, and right EEF position/rot6d 23–31.
For the released camera-EEF policy, EEF states and absolute targets must already
be expressed in the ROS optical head-camera frame. Do not feed world-frame EEF
values into this recipe. The backend then forms anchor-relative deltas.

For a different robot, configure source selection, destination slots and valid
dimensions independently. Do not copy the RoboDojo mapping onto a dataset with
different field ordering. The canonical 34D pretraining example already expects
34D fields and therefore does not apply the 32D remapping.

`chunk_delta` subtracts the anchor state from selected translation/joint slots.
Rotation groups use `R_target @ R_reference.T` and return rot6d; grippers stay
absolute. Units, reference frame, quaternion/rotation conventions and gripper
range must be known before conversion. A matching vector width alone is not
sufficient compatibility.

History is strictly past-only. The SFT recipe uses 25 past observations with an
interval of 20 action steps, pooled into 16 tokens per valid frame. The reader
and history adapter account for source action upsampling; unavailable history
at episode start is omitted. Future frames are teacher targets, not inference
inputs.

## Normalization and checks

Generate statistics with the same transforms as the training recipe:

```bash
python scripts/compute_norm_stats.py \
  --config configs/train/robodojo_sft.yaml --dataset-index 0 \
  --output .cache/magic_w0/norm/robodojo.json

python scripts/check_data.py --config configs/train/robodojo_sft.yaml --samples 2
```

For the pretraining group, select each child source with `--source-index`:

```bash
python scripts/compute_norm_stats.py \
  --config configs/train/pretrain.yaml --dataset-index 0 --source-index 0 \
  --output .cache/magic_w0/norm/robot_a.json
```

Repeat for source 1 and `robot_b.json`. If you relocate the cache root, use that
root in the output argument too. Statistics include a dataset/transform
signature. Regenerate them after changing action mappings, rotation groups,
chunk size, source selection or relevant filters. The examples reject
dataset-signature mismatches rather than silently reusing another source's
statistics.

`check_data.py --list` only lists enabled sources. Without `--list`, it opens
the real reader and model data processor, so missing fields, masks, frames and
normalization artifacts fail before allocating a model on a GPU.
