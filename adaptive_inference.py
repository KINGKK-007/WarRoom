"""
adaptive_inference.py — broader observation-driven adaptive baseline for DevOps War Room
"""

import os
import sys
from typing import Any, Dict, List

from baseline_policy import (
    BASELINE_SEED,
    ENV_URL,
    MAX_EPISODE_STEPS,
    REQUEST_TIMEOUT_S,
    TASK_MAPPING,
    TASKS,
    build_session,
    choose_action,
    grade_task,
    log_end,
    log_start,
    log_step,
    structured_action,
    success,
)


MODEL_NAME = os.environ.get("MODEL_NAME", "adaptive-baseline")
CURRENT_SESSION_ID: str | None = None
SESSION = build_session()


def _post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    response = SESSION.post(f"{ENV_URL}{path}", json=payload, timeout=REQUEST_TIMEOUT_S)
    response.raise_for_status()
    return response.json()


def _get(path: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    response = SESSION.get(f"{ENV_URL}{path}", params=params, timeout=REQUEST_TIMEOUT_S)
    response.raise_for_status()
    return response.json()


def reset_env(task_id: str, seed: int | None = None) -> Dict[str, Any]:
    global CURRENT_SESSION_ID
    scenario = TASK_MAPPING.get(task_id, task_id)
    payload: Dict[str, Any] = {"task_id": scenario}
    if seed is not None:
        payload["seed"] = seed
    body = _post("/reset", payload)
    CURRENT_SESSION_ID = body["session_id"]
    return body["observation"]


def step_env(action: str) -> Dict[str, Any]:
    if CURRENT_SESSION_ID is None:
        raise RuntimeError("No active session. Call reset_env() before step_env().")
    return _post("/step", {"session_id": CURRENT_SESSION_ID, "action_type": "raw_command", "params": {"command": action}})


def state_env() -> Dict[str, Any]:
    if CURRENT_SESSION_ID is None:
        raise RuntimeError("No active session. Call reset_env() before state_env().")
    return _get("/state", {"session_id": CURRENT_SESSION_ID})["state"]


def run_task(task_id: str, seed: int | None = None) -> float:
    effective_seed = BASELINE_SEED if seed is None else seed
    log_start(task_id, ENV_URL, MODEL_NAME)
    observation = reset_env(task_id, seed=effective_seed)
    command_history: List[str] = []
    action_history: List[Dict[str, Any]] = []
    total_reward = 0.0
    step_num = 0

    while step_num < MAX_EPISODE_STEPS:
        step_num += 1
        state = state_env()
        action = choose_action(observation, state, command_history, adaptive=True)
        result = step_env(action)
        reward = float((result.get("reward") or {}).get("value", 0.0))
        observation = result.get("observation") or observation
        done = bool(result.get("done", False))
        info = result.get("info") or {}

        command_history.append(action)
        action_history.append(structured_action(action))
        total_reward += reward
        log_step(step_num, action, reward, done, info.get("error"))

        if done:
            break

    final_state = state_env()
    score = grade_task(task_id, action_history, final_state)
    log_end(success(final_state, score), step_num, score, total_reward)
    return score


if __name__ == "__main__":
    task_arg = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "--task" else None
    tasks = [task_arg] if task_arg else TASKS
    for task in tasks:
        run_task(task)
