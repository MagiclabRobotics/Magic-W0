"""Magic-W0 inference components; tensor names and computations are preserved."""

from __future__ import annotations
from typing import Any
import torch
from torch import nn
from .config import (
    ACTION_DIM_MASK,
    CAMERA_VALID,
    EMBODIMENT,
    HISTORY_FRAMES,
    HISTORY_IMAGES,
    HISTORY_LEN,
    HISTORY_PLACEHOLDER_TOKEN,
    MagicW0Config,
    STATE,
    STATE_DIM_MASK,
    TASK,
    VIEW_VALID_MASK,
    WORLD_3D_VALID_MASK,
    describe,
)
from .modules import Hybrid2DWorldExpert, Hybrid3DWorldExpert, HybridActionExpert
from .processing import (
    _maybe_to,
    _render_robot_prompt_metadata,
    history_block_text,
    image_prompt_label,
    pool_frame_features,
)


class MagicW0(nn.Module):
    """Inference runtime for the Magic_W0 history checkpoint."""

    def __init__(
        self, config: MagicW0Config, *, vlm_state_dict: dict[str, torch.Tensor]
    ) -> None:
        super().__init__()
        self.config = config
        checkpoint = config.validate_checkpoint_layout()
        self._validate_runtime_version()
        from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

        hf_config = AutoConfig.from_pretrained(
            checkpoint, local_files_only=True, trust_remote_code=False
        )
        if hf_config.model_type != config.vlm_model_type:
            raise ValueError(
                f"Magic_W0 requires model_type={config.vlm_model_type!r}, got {hf_config.model_type!r} from {checkpoint}"
            )
        text_config = getattr(hf_config, "text_config", hf_config)
        self._validate_hybrid_layout(config, text_config)
        self.vlm_layer_types = tuple(text_config.layer_types)
        self.processor = AutoProcessor.from_pretrained(
            checkpoint, local_files_only=True, trust_remote_code=False
        )
        construction_kwargs = {
            "dtype": self._torch_dtype(config.dtype),
            "attn_implementation": "sdpa",
        }
        with torch.device("meta"):
            self.vlm = AutoModelForImageTextToText.from_config(
                hf_config, **construction_kwargs
            )
        self.vlm.load_state_dict(vlm_state_dict, strict=True, assign=True)
        visual_rotary = self.vlm.model.visual.rotary_pos_emb
        self.vlm.model.visual.rotary_pos_emb = type(visual_rotary)(
            visual_rotary.dim, visual_rotary.theta
        )
        text_rotary = self.vlm.model.language_model.rotary_emb
        self.vlm.model.language_model.rotary_emb = type(text_rotary)(text_config)
        unmaterialized = [
            name
            for name, value in list(self.vlm.named_parameters())
            + list(self.vlm.named_buffers())
            if value.is_meta
        ]
        if unmaterialized:
            raise RuntimeError(
                f"VLM checkpoint left meta tensors unmaterialized: {unmaterialized}"
            )
        self.vlm.config.use_cache = False
        self._lm_head = self.vlm.get_output_embeddings()
        self.action_expert = HybridActionExpert(config)
        self.action_expert.to(dtype=self._torch_dtype(config.dtype))
        vlm_hidden = int(self.vlm.config.text_config.hidden_size)
        self.state_prompt_projection = nn.Sequential(
            nn.Linear(config.state_dim * 2, vlm_hidden),
            nn.SiLU(),
            nn.Linear(vlm_hidden, vlm_hidden * config.state_prompt_tokens),
        )
        self.state_prompt_projection.to(dtype=self._torch_dtype(config.dtype))
        for module in self.state_prompt_projection:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)
        dtype = self._torch_dtype(config.dtype)
        self.action_residual_dropout = nn.Identity()
        self.world_2d_expert = Hybrid2DWorldExpert(
            config, config.dinov3_target_hidden_size
        ).to(dtype=dtype)
        self.world_3d_expert = Hybrid3DWorldExpert(config).to(dtype=dtype)
        self.requires_grad_(False)
        self.eval()

    @staticmethod
    def _validate_hybrid_layout(config: MagicW0Config, text_config: Any) -> None:
        layer_types = list(text_config.layer_types)
        expected = [
            "full_attention"
            if (index + 1) % config.full_attention_interval == 0
            else "linear_attention"
            for index in range(config.action_expert_num_layers)
        ]
        if int(text_config.num_hidden_layers) != config.action_expert_num_layers:
            raise ValueError(
                f"VLM and action expert must have equal depth: {text_config.num_hidden_layers} != {config.action_expert_num_layers}"
            )
        if layer_types != expected:
            raise ValueError(
                "VLM layer schedule does not match the configured 3:1 hybrid expert"
            )
        checks = {
            "num_attention_heads": config.full_attn_num_heads,
            "num_key_value_heads": config.full_attn_num_kv_heads,
            "head_dim": config.full_attn_head_dim,
            "linear_num_key_heads": config.deltanet_num_key_heads,
            "linear_num_value_heads": config.deltanet_num_value_heads,
            "linear_key_head_dim": config.deltanet_key_head_dim,
            "linear_value_head_dim": config.deltanet_value_head_dim,
            "linear_conv_kernel_dim": config.deltanet_conv_kernel,
        }
        mismatches = {
            name: (int(getattr(text_config, name)), expected_value)
            for name, expected_value in checks.items()
            if int(getattr(text_config, name)) != expected_value
        }
        if mismatches:
            raise ValueError(
                f"expert attention dimensions must mirror Qwen3.5 for joint attention; mismatches={mismatches}"
            )

    @staticmethod
    def _validate_runtime_version() -> None:
        try:
            import transformers
            from packaging.version import Version
        except ImportError as exc:
            raise ImportError("Magic_W0 requires transformers>=5.1.0") from exc
        if Version(transformers.__version__) < Version("5.1.0"):
            raise RuntimeError(
                f"Qwen3.5 requires transformers>=5.1.0; found {transformers.__version__}"
            )

    @staticmethod
    def _torch_dtype(name: str) -> torch.dtype:
        return torch.bfloat16 if name == "bfloat16" else torch.float32

    def _state_placeholder_id(self) -> int:
        """Reserve the last vocabulary id for non-text state embeddings."""
        vocab = int(self.vlm.get_input_embeddings().weight.shape[0])
        return vocab - 1

    def _append_state_placeholders(self, vlm_inputs: dict[str, Any]) -> dict[str, Any]:
        """Append state slots after the causal image and task prefix."""
        count = int(self.config.state_prompt_tokens)
        input_ids = vlm_inputs["input_ids"]
        prefix_len = int(input_ids.shape[1])
        batch_size = int(input_ids.shape[0])
        placeholder = torch.full(
            (batch_size, count), self._state_placeholder_id(), dtype=input_ids.dtype
        )
        updated = dict(vlm_inputs)
        for key, value in vlm_inputs.items():
            if not (
                torch.is_tensor(value)
                and value.ndim == 2
                and (value.shape[1] == prefix_len)
            ):
                continue
            if key == "input_ids":
                tail = placeholder
            elif key == "attention_mask":
                tail = torch.ones(batch_size, count, dtype=value.dtype)
            else:
                tail = torch.zeros(batch_size, count, dtype=value.dtype)
            updated[key] = torch.cat([value, tail], dim=1)
        return updated

    def _prefix_text_parts(
        self,
        sample_index,
        states,
        tasks,
        embodiments=None,
        state_dim_masks=None,
        action_dim_masks=None,
    ) -> tuple[str, str]:
        """Render the embodiment, task, and action prediction prompt."""
        state_text = ""
        closing = "Predict the next robot action chunk given the current robot state:"
        description = describe(embodiments[sample_index]) if embodiments else None
        embodiment_text = f"Robot embodiment: {description}\n" if description else ""
        prompt_metadata = _render_robot_prompt_metadata(
            self.config.robot_prompt_metadata,
            None if state_dim_masks is None else state_dim_masks[sample_index],
            None if action_dim_masks is None else action_dim_masks[sample_index],
        )
        metadata_text = f"{prompt_metadata}\n" if prompt_metadata else ""
        return (
            f"{embodiment_text}{metadata_text}Robot task: {tasks[sample_index]}\n",
            f"{state_text}{closing}",
        )

    def _finalize_robotics_prefix(self, messages):
        """Left-pad the image/task prefix, then append state embedding slots."""
        tokenizer = getattr(self.processor, "tokenizer", None)
        previous_padding_side = getattr(tokenizer, "padding_side", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left"
        try:
            vlm_inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
                processor_kwargs={"padding": True},
            )
        finally:
            if tokenizer is not None and previous_padding_side is not None:
                tokenizer.padding_side = previous_padding_side
        vlm_inputs = self._append_state_placeholders(vlm_inputs)
        return vlm_inputs, int(vlm_inputs["input_ids"].shape[1])

    def _prepare_robotics_prefix(
        self,
        batch,
        states,
        tasks,
        camera_valid,
        embodiments=None,
        state_dim_masks=None,
        action_dim_masks=None,
    ):
        """Build the Magic_W0 image-history prefix, including empty episode history.

        ``task line + History images: <past frames>`` ->
        ``Head image: <img> Left wrist image: <img> ...`` -> state + closing.
        """
        if HISTORY_FRAMES not in batch or HISTORY_LEN not in batch:
            raise KeyError(
                f"history_enabled=True but the batch carries no {HISTORY_FRAMES!r}/{HISTORY_LEN!r}; build them with the adapter history buffer"
            )
        clips, lengths = (batch[HISTORY_FRAMES], batch[HISTORY_LEN])
        expected_shape = (
            int(states.shape[0]),
            self.config.history_frames,
            3,
            self.config.image_size,
            self.config.image_size,
        )
        if not torch.is_tensor(clips) or tuple(clips.shape) != expected_shape:
            raise ValueError(f"history_frames must have shape {expected_shape}")
        if (
            not torch.is_tensor(lengths)
            or tuple(lengths.shape) != (expected_shape[0],)
            or lengths.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("history_len must be an integer [B] tensor")
        if bool(((lengths < 0) | (lengths > expected_shape[1])).any()):
            raise ValueError("history_len must be between zero and history_frames")
        messages = []
        past_frames: list[torch.Tensor] = []
        for sample_index in range(int(states.shape[0])):
            length = int(batch[HISTORY_LEN][sample_index])
            if length:
                past_frames.extend(
                    batch[HISTORY_FRAMES][sample_index][-length:].detach().cpu()
                )
            task_text, tail_text = self._prefix_text_parts(
                sample_index,
                states,
                tasks,
                embodiments,
                state_dim_masks,
                action_dim_masks,
            )
            content: list[dict[str, Any]] = [
                {
                    "type": "text",
                    "text": task_text
                    + history_block_text(length, self.config.history_tokens_per_frame),
                }
            ]
            for camera_index, camera_key in enumerate(self.config.camera_keys):
                if not bool(camera_valid[sample_index, camera_index]):
                    continue
                image = batch[camera_key][sample_index]
                if image.ndim == 4:
                    image = image[-1]
                content.append(
                    {
                        "type": "text",
                        "text": f"{image_prompt_label(camera_key)} image: ",
                    }
                )
                content.append({"type": "image", "image": image.detach().cpu()})
            content.append({"type": "text", "text": tail_text})
            messages.append([{"role": "user", "content": content}])
        vlm_inputs, prefix_len = self._finalize_robotics_prefix(messages)
        if past_frames:
            encoded = self.processor.image_processor(
                images=past_frames, return_tensors="pt"
            )
            pixel_values = encoded["pixel_values"]
            if self.config.dtype == "bfloat16":
                pixel_values = pixel_values.to(torch.bfloat16)
            vlm_inputs = dict(vlm_inputs)
            vlm_inputs[HISTORY_IMAGES] = {
                "pixel_values": pixel_values,
                "image_grid_thw": encoded["image_grid_thw"],
            }
        return vlm_inputs, prefix_len

    def _history_placeholder_id(self) -> int:
        cached = getattr(self, "_history_placeholder_cache", None)
        if cached is None:
            cached = int(
                self.processor.tokenizer.convert_tokens_to_ids(
                    HISTORY_PLACEHOLDER_TOKEN
                )
            )
            self._history_placeholder_cache = cached
        return cached

    def prepare_inference_batch(
        self, batch: dict[str, Any], device: torch.device | None
    ) -> dict[str, Any]:
        """Build a robotics prefix without actions or future teacher targets.

        Images follow ``config.camera_keys`` and may be uint8 CHW or a temporal
        stack. State normalization remains the caller's responsibility, matching
        the training processor contract.
        """
        if STATE not in batch or TASK not in batch:
            raise ValueError(f"inference batch requires {STATE!r} and {TASK!r}")
        states = batch[STATE]
        if not torch.is_tensor(states) or states.ndim != 2:
            raise ValueError("inference state must be a [B,state_dim] tensor")
        if states.shape[1] != self.config.state_dim:
            raise ValueError(
                f"inference state width {states.shape[1]} != {self.config.state_dim}"
            )
        batch_size = int(states.shape[0])
        camera_valid = batch.get(CAMERA_VALID)
        if camera_valid is None:
            camera_valid = torch.ones(
                (batch_size, len(self.config.camera_keys)), dtype=torch.bool
            )
        if tuple(camera_valid.shape) != (batch_size, len(self.config.camera_keys)):
            raise ValueError(
                f"camera_valid must be [B,num_configured_cameras], got {tuple(camera_valid.shape)}"
            )
        missing_images = [
            key
            for index, key in enumerate(self.config.camera_keys)
            if bool(camera_valid[:, index].any()) and key not in batch
        ]
        if missing_images:
            raise ValueError(
                f"inference batch is missing camera images {missing_images}"
            )
        state_dim_mask = batch.get(STATE_DIM_MASK)
        if state_dim_mask is not None and tuple(state_dim_mask.shape) != tuple(
            states.shape
        ):
            raise ValueError("state_dim_mask must have the same shape as state")
        vlm_inputs, prefix_len = self._prepare_robotics_prefix(
            batch,
            states,
            batch[TASK],
            camera_valid,
            batch.get(EMBODIMENT),
            state_dim_mask,
            action_dim_masks=batch.get(ACTION_DIM_MASK),
        )
        prepared: dict[str, Any] = {
            "prefix_len": prefix_len,
            "vlm_inputs": {
                key: _maybe_to(value, device) if torch.is_tensor(value) else value
                for key, value in vlm_inputs.items()
            },
            STATE: _maybe_to(states, device),
        }
        if state_dim_mask is not None:
            prepared[STATE_DIM_MASK] = _maybe_to(state_dim_mask, device)
        camera_index = {key: index for index, key in enumerate(self.config.camera_keys)}
        view_mask = torch.zeros(
            (batch_size, self.config.gen_2d_max_views), dtype=torch.bool
        )
        for view_index, key in enumerate(self.config.gen_2d_camera_keys):
            if key not in camera_index:
                raise ValueError(f"2D camera {key!r} is absent from camera_keys")
            view_mask[:, view_index] = camera_valid[:, camera_index[key]].bool()
        prepared[VIEW_VALID_MASK] = _maybe_to(view_mask, device)
        key = self.config.gen_3d_camera_key
        if key not in camera_index:
            raise ValueError(f"3D camera {key!r} is absent from camera_keys")
        prepared[WORLD_3D_VALID_MASK] = _maybe_to(
            camera_valid[:, camera_index[key]].bool(), device
        )
        return prepared

    @torch.no_grad()
    def generate_actions(
        self,
        batch: dict[str, Any],
        num_steps: int | None = None,
        initial_noise: torch.Tensor | None = None,
        action_dim_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Integrate the joint flow field from noise (t=1) to actions (t=0)."""
        if "vlm_inputs" not in batch or STATE not in batch:
            raise ValueError("generate_actions expects prepare_inference_batch output")
        n_steps = int(
            self.config.num_inference_steps if num_steps is None else num_steps
        )
        if n_steps <= 0:
            raise ValueError("num_steps must be positive")
        states = batch[STATE]
        expected = (states.shape[0], self.config.chunk_size, self.config.action_dim)
        if initial_noise is None:
            actions = torch.randn(expected, device=states.device, dtype=states.dtype)
        else:
            if tuple(initial_noise.shape) != expected:
                raise ValueError(
                    f"initial_noise shape {tuple(initial_noise.shape)} != {expected}"
                )
            actions = initial_noise.to(device=states.device, dtype=states.dtype).clone()
        valid_dimensions = None
        if action_dim_mask is None:
            action_attention_mask = torch.ones(
                expected[:2], dtype=torch.bool, device=states.device
            )
        else:
            raw_mask = torch.as_tensor(
                action_dim_mask, dtype=torch.bool, device=states.device
            )
            if raw_mask.ndim == 1 and tuple(raw_mask.shape) == (expected[2],):
                valid_dimensions = raw_mask[None, None, :].expand(expected)
            elif raw_mask.ndim == 2 and tuple(raw_mask.shape) == (
                expected[0],
                expected[2],
            ):
                valid_dimensions = raw_mask[:, None, :].expand(expected)
            elif raw_mask.ndim == 3 and tuple(raw_mask.shape) == expected:
                valid_dimensions = raw_mask
            else:
                raise ValueError(
                    f"action_dim_mask must be [D], [B,D], or [B,T,D], got {tuple(raw_mask.shape)} for actions {expected}"
                )
            action_attention_mask = valid_dimensions.any(dim=-1)
            actions = actions.masked_fill(~valid_dimensions, 0.0)
        dt = 1.0 / n_steps
        self.eval()
        flow_cache = self._prepare_flow_inference_cache(batch)
        for step in range(n_steps):
            timestep = torch.full(
                (expected[0],),
                1.0 - step * dt,
                device=states.device,
                dtype=torch.float32,
            )
            velocity = self._run_cached_joint_expert_trunks(
                actions, timestep, states, action_attention_mask, flow_cache
            )
            actions = actions - dt * velocity.to(actions.dtype)
            if valid_dimensions is not None:
                actions = actions.masked_fill(~valid_dimensions, 0.0)
        return actions

    def _active_joint_expert_components(
        self,
    ) -> list[tuple[str, nn.ModuleList, nn.Dropout]]:
        """Return enabled expert stacks in their Joint Attention stream order."""
        components: list[tuple[str, nn.ModuleList, nn.Dropout]] = []
        components.append(
            (
                "world_2d",
                self.world_2d_expert.blocks,
                self.world_2d_expert.residual_dropout,
            )
        )
        components.append(
            (
                "world_3d",
                self.world_3d_expert.blocks,
                self.world_3d_expert.residual_dropout,
            )
        )
        components.append(
            ("action", self.action_expert.blocks, self.action_residual_dropout)
        )
        return components

    def _prepare_flow_inference_cache(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Run the action-independent VLM branch once for one flow trajectory.

        Joint Attention is asymmetric: every expert reads the VLM prefix, while
        the VLM never reads world/action tokens. Consequently the VLM hidden
        states and its projected per-layer K/V are identical for all reverse-flow
        steps. World experts are intentionally not cached because their full
        attention reads the changing noisy Action stream.
        """
        states = batch[STATE]
        vlm_hidden, vlm_padding_mask, vlm_position_ids = self._embed_vlm_inputs(
            batch["vlm_inputs"], states, batch.get(STATE_DIM_MASK)
        )
        language_model = self.vlm.model.language_model
        vlm_position_embeddings = language_model.rotary_emb(
            vlm_hidden, vlm_position_ids
        )
        from transformers.masking_utils import create_causal_mask

        vlm_causal_mask = create_causal_mask(
            config=language_model.config,
            inputs_embeds=vlm_hidden,
            attention_mask=vlm_padding_mask,
            past_key_values=None,
            position_ids=vlm_position_ids[0],
        )
        prefix_len = int(batch["prefix_len"])
        prefix_mask = vlm_padding_mask[:, :prefix_len]
        prefix_position_ids = vlm_position_ids[:, :, :prefix_len]
        prefix_rope = tuple(
            (value[:, :prefix_len] for value in vlm_position_embeddings)
        )
        valid_prefix_positions = prefix_position_ids.masked_fill(
            ~prefix_mask.unsqueeze(0), 0
        )
        world_start = valid_prefix_positions.amax(dim=(0, 2)).add(1)
        world_hidden_states: list[torch.Tensor] = []
        world_masks: list[torch.Tensor] = []
        world_position_embeddings: list[tuple[torch.Tensor, torch.Tensor]] = []
        max_world_len = 0
        view_valid_mask = batch.get(VIEW_VALID_MASK)
        if view_valid_mask is None:
            raise ValueError("2D stream requires view_valid_mask")
        world_2d = self.world_2d_expert.embed_queries(states.shape[0], states.device)
        positions_2d = world_start[:, None] + torch.arange(
            world_2d.shape[1], device=states.device
        )
        world_hidden_states.append(world_2d)
        world_masks.append(self._world_2d_mask(view_valid_mask))
        world_position_embeddings.append(
            language_model.rotary_emb(
                world_2d, positions_2d.unsqueeze(0).expand(3, -1, -1)
            )
        )
        max_world_len = max(max_world_len, world_2d.shape[1])
        world_3d_valid_mask = batch.get(WORLD_3D_VALID_MASK)
        if world_3d_valid_mask is None:
            raise ValueError("3D stream requires world_3d_valid_mask")
        world_3d = self.world_3d_expert.embed_queries(states.shape[0], states.device)
        positions_3d = world_start[:, None] + torch.arange(
            world_3d.shape[1], device=states.device
        )
        world_hidden_states.append(world_3d)
        world_masks.append(self._world_3d_mask(world_3d_valid_mask))
        world_position_embeddings.append(
            language_model.rotary_emb(
                world_3d, positions_3d.unsqueeze(0).expand(3, -1, -1)
            )
        )
        max_world_len = max(max_world_len, world_3d.shape[1])
        action_start = world_start + max_world_len
        action_positions = action_start[:, None] + torch.arange(
            self.config.chunk_size, device=states.device
        )
        action_rope_reference = vlm_hidden.new_empty(
            (
                states.shape[0],
                self.config.chunk_size,
                self.config.action_expert_hidden_size,
            )
        )
        action_position_embeddings = language_model.rotary_emb(
            action_rope_reference, action_positions.unsqueeze(0).expand(3, -1, -1)
        )
        components = self._active_joint_expert_components()
        if any(
            (len(blocks) != len(language_model.layers) for _, blocks, _ in components)
        ):
            raise RuntimeError("VLM and joint expert layer counts differ")
        vlm_key_values: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        for layer_index, vlm_layer in enumerate(language_model.layers):
            blocks = [component[1][layer_index] for component in components]
            is_full = blocks[0].use_full_attention
            if any((block.use_full_attention != is_full for block in blocks)):
                raise RuntimeError("joint expert full-attention schedules diverged")
            if is_full:
                vlm_context = vlm_layer.input_layernorm(vlm_hidden)[:, :prefix_len]
                vlm_key_values.append(
                    blocks[0].mixer._vlm_kv(
                        vlm_context, vlm_layer.self_attn, prefix_rope
                    )
                )
                attention_mask = vlm_causal_mask
            else:
                vlm_key_values.append(None)
                attention_mask = vlm_padding_mask
            vlm_hidden = vlm_layer(
                vlm_hidden,
                position_embeddings=vlm_position_embeddings,
                attention_mask=attention_mask,
                position_ids=vlm_position_ids[0],
                past_key_values=None,
                use_cache=False,
            )
        return {
            "stream_names": tuple((name for name, _, _ in components)),
            "world_hidden_states": tuple(world_hidden_states),
            "world_masks": tuple(world_masks),
            "world_position_embeddings": tuple(world_position_embeddings),
            "action_position_embeddings": action_position_embeddings,
            "prefix_mask": prefix_mask,
            "vlm_key_values": tuple(vlm_key_values),
        }

    def _world_2d_mask(self, view_mask: torch.Tensor) -> torch.Tensor:
        return view_mask.bool().repeat_interleave(
            self.config.gen_2d_num_query_tokens_per_view, dim=1
        )

    def _world_3d_mask(self, sample_mask: torch.Tensor) -> torch.Tensor:
        return (
            sample_mask.bool()
            .unsqueeze(1)
            .expand(-1, self.config.gen_3d_num_query_tokens)
        )

    def _embed_vlm_inputs(
        self,
        vlm_inputs: dict[str, Any],
        states: torch.Tensor | None = None,
        state_dim_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        multimodal_model = self.vlm.model
        input_ids = vlm_inputs["input_ids"]
        attention_mask = vlm_inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        inputs_embeds = multimodal_model.get_input_embeddings()(input_ids)
        if self.state_prompt_projection is not None and states is not None:
            if state_dim_mask is None:
                state_dim_mask = torch.ones_like(states, dtype=torch.bool)
            features = torch.cat((states, state_dim_mask.to(states.dtype)), dim=-1).to(
                inputs_embeds.dtype
            )
            state_embeds = self.state_prompt_projection(features).reshape(
                states.shape[0], int(self.config.state_prompt_tokens), -1
            )
            state_mask = (input_ids == self._state_placeholder_id()).unsqueeze(-1)
            inputs_embeds = inputs_embeds.masked_scatter(
                state_mask, state_embeds.to(inputs_embeds.dtype)
            )
        pixel_values = vlm_inputs.get("pixel_values")
        image_grid_thw = vlm_inputs.get("image_grid_thw")
        if pixel_values is not None:
            image_outputs = multimodal_model.get_image_features(
                pixel_values, image_grid_thw, return_dict=True
            )
            image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            image_mask, _ = multimodal_model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        position_ids = multimodal_model.compute_3d_position_ids(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            attention_mask=attention_mask,
            past_key_values=None,
            mm_token_type_ids=vlm_inputs.get("mm_token_type_ids"),
        )
        if position_ids is None:
            text_positions = attention_mask.long().cumsum(dim=-1).sub(1).clamp_min(0)
            position_ids = text_positions.unsqueeze(0).expand(3, -1, -1)
        placeholder = vlm_inputs["input_ids"] == self._history_placeholder_id()
        expected = int(placeholder.sum())
        frames = vlm_inputs.get(HISTORY_IMAGES)
        if frames is None:
            if expected:
                raise ValueError(
                    f"{expected} history placeholders but no history frames"
                )
            return (inputs_embeds, attention_mask, position_ids)
        device = inputs_embeds.device
        grid_thw = frames["image_grid_thw"].to(device)
        outputs = self.vlm.model.get_image_features(
            frames["pixel_values"].to(device), grid_thw, return_dict=True
        )
        pooled = pool_frame_features(
            list(outputs.pooler_output),
            grid_thw,
            merge=int(self.vlm.config.vision_config.spatial_merge_size),
            pool_size=int(self.config.history_pool_size),
        )
        if int(pooled.shape[0]) != expected:
            raise ValueError(
                f"{expected} history placeholders but {int(pooled.shape[0])} pooled tokens"
            )
        inputs_embeds = inputs_embeds.masked_scatter(
            placeholder.to(device).unsqueeze(-1), pooled.to(device, inputs_embeds.dtype)
        )
        return (inputs_embeds, attention_mask, position_ids)

    def _run_cached_joint_expert_trunks(
        self, noisy_actions, timesteps, states, action_attention_mask, cache
    ):
        """Every world/action query reads the VLM prefix and all three expert streams."""
        components = self._active_joint_expert_components()
        hidden_states = list(cache["world_hidden_states"])
        hidden_states.append(
            self.action_expert.embed_inputs(noisy_actions, timesteps, states)
        )
        masks = [*cache["world_masks"], action_attention_mask.bool()]
        ropes = [
            *cache["world_position_embeddings"],
            cache["action_position_embeddings"],
        ]
        for layer_index, vlm_kv in enumerate(cache["vlm_key_values"]):
            blocks = [component[1][layer_index] for component in components]
            if not blocks[0].use_full_attention:
                updated = []
                for hidden, mask, block in zip(
                    hidden_states, masks, blocks, strict=True
                ):
                    hidden = hidden + block.mixer(block.input_layernorm(hidden), mask)
                    hidden = hidden + block.mlp(block.post_attention_layernorm(hidden))
                    updated.append(hidden)
                hidden_states = updated
                continue
            if vlm_kv is None:
                raise RuntimeError("missing VLM K/V for full-attention layer")
            qkv = [
                block.mixer._expert_qkv(block.input_layernorm(hidden), rope)
                for block, hidden, rope in zip(
                    blocks, hidden_states, ropes, strict=True
                )
            ]
            key = torch.cat([vlm_kv[0], *(x[1] for x in qkv)], dim=2)
            value = torch.cat([vlm_kv[1], *(x[2] for x in qkv)], dim=2)
            key_mask = torch.cat((cache["prefix_mask"], *masks), dim=1)
            mixer = blocks[0].mixer
            prepared = mixer.prepare_attention_kv(key, value, key_mask)
            updated = []
            for hidden, block, (query, _, _, gate) in zip(
                hidden_states, blocks, qkv, strict=True
            ):
                attended = mixer.attend_prepared(query, prepared)
                attended = attended.transpose(1, 2).reshape(
                    hidden.shape[0], hidden.shape[1], -1
                )
                hidden = hidden + block.mixer.o_proj(attended * torch.sigmoid(gate))
                hidden = hidden + block.mlp(block.post_attention_layernorm(hidden))
                updated.append(hidden)
            hidden_states = updated
        return self.action_expert.project_output(hidden_states[-1])
