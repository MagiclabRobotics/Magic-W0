"""XPolicyLab evaluation loop for the inference-only Magic_W0 policy."""

from .robodojo_adapter import (
    head_camera_to_env,
    prepare_action,
    prepare_observation,
)


def _observations(task_env, env_indices):
    return [
        prepare_observation(task_env, obs)
        for obs in task_env.get_obs_batch(env_indices)
    ]


def _planning_observations(task_env, observations):
    # Chunk-local state: reset/replan and changing active indices cannot reuse it.
    transforms = {}
    prepared = []
    for observation in observations:
        env_idx = observation.get("env_idx", 0)
        transform = (
            head_camera_to_env(task_env, env_idx)
            if hasattr(task_env, "camera_manager")
            else None
        )
        transforms[env_idx] = transform
        prepared.append(
            prepare_observation(task_env, observation, camera_to_env=transform)
        )
    return prepared, transforms


def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    while not TASK_ENV.is_episode_end():
        observations, transforms = _planning_observations(
            TASK_ENV, [TASK_ENV.get_obs()]
        )
        env_idx = observations[0].get("env_idx", 0)
        model_client.call(func_name="update_obs", obs=observations[0])
        actions = model_client.call(func_name="get_action")
        for action_index, action in enumerate(actions):
            TASK_ENV.take_action(
                prepare_action(
                    TASK_ENV, action, env_idx, camera_to_env=transforms[env_idx]
                )
            )
            if TASK_ENV.is_episode_end() or action_index + 1 == len(actions):
                break
            model_client.call(
                func_name="update_obs",
                obs=prepare_observation(TASK_ENV, TASK_ENV.get_obs()),
            )


def eval_one_episode_batch(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    while not TASK_ENV.is_episode_end():
        env_indices = TASK_ENV.get_running_env_idx_list()
        observations, transforms = _planning_observations(
            TASK_ENV, TASK_ENV.get_obs_batch(env_indices)
        )
        model_client.call(
            func_name="update_obs_batch",
            obs=observations,
        )
        actions = model_client.call(func_name="get_action_batch", obs=env_indices)
        chunk_size = len(actions[0])
        for action_index in range(chunk_size):
            TASK_ENV.take_action_batch(
                [
                    prepare_action(
                        TASK_ENV,
                        env_actions[action_index],
                        env_idx,
                        camera_to_env=transforms[env_idx],
                    )
                    for env_actions, env_idx in zip(actions, env_indices, strict=True)
                ],
                env_indices,
            )
            if TASK_ENV.is_episode_end() or action_index + 1 == chunk_size:
                break
            running = set(TASK_ENV.get_running_env_idx_list())
            active = [
                i for i, env_index in enumerate(env_indices) if env_index in running
            ]
            actions = [actions[i] for i in active]
            env_indices = [env_indices[i] for i in active]
            model_client.call(
                func_name="update_obs_batch",
                obs=_observations(TASK_ENV, env_indices),
            )
