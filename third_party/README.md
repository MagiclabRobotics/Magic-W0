# Track4World training adapter

The MagicW0 teacher uses geometry/motion feature taps, local DA3/ConvNeXt loading,
bounded motion batches and per-sample depth alignment. These additions are
provided as `track4world-magicw0.patch`, with an exact public source revision and
file hashes in `track4world.json`.

```bash
python scripts/prepare_track4world.py
python scripts/prepare_track4world.py --check
```

Preparation downloads the pinned public Track4World and utils3d archives,
verifies their SHA-256 checksums, applies the patch, and verifies all modified
source hashes. It refuses to overwrite a different existing source tree. Use
`--output-dir` and update the training YAML for a custom destination. Offline
preparation accepts `--archive` and `--utils3d-archive`.

Teacher weights remain separate resources. The prepared source directory is
ignored by Git. Its upstream license is retained; the patch is subject to the
same applicable upstream terms. See `LICENSES/Track4World.txt` and the upstream
source notices, including the vendored Depth Anything 3 Apache-2.0 notices.
The repository's MIT license does not replace these terms.
