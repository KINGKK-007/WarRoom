"""
inference.py — deterministic observation-driven baseline agent for DevOps War Room
"""

import os
import sys
from typing import Any, Dict, List

from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://coolalien35-warroom-deploy.hf.space")
MODEL_NAME = os.getenv("MODEL_NAME", "deterministic-baseline")
HF_TOKEN = os.getenv("HF_TOKEN")

from baseline_policy import (
    BASELINE_SEED,
    MAX_EPISODE_STEPS,
    REQUEST_TIMEOUT_S,
    TASK_MAPPING,
    TASKS,
    build_session,
    choose_action,
    grade_task,
    structured_action,
)
CURRENT_SESSION_ID: str | None = None
SESSION = build_session()


def _post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    response = SESSION.post(f"{API_BASE_URL.rstrip('/')}{path}", json=payload, timeout=REQUEST_TIMEOUT_S)
    response.raise_for_status()
    return response.json()


def _get(path: str) -> Dict[str, Any]:
    response = SESSION.get(f"{API_BASE_URL.rstrip('/')}{path}", timeout=REQUEST_TIMEOUT_S)
    response.raise_for_status()
    return response.json()


def reset_env(task_id: str, seed: int | None = None) -> Dict[str, Any]:
    global CURRENT_SESSION_ID
    scenario = TASK_MAPPING.get(task_id, task_id)
    payload: Dict[str, Any] = {"task_id": scenario}
    if seed is not None:
        payload["seed"] = seed
    data = _post("/reset", payload)
    CURRENT_SESSION_ID = data["session_id"]
    return data["observation"]


def step_env(action: str) -> Dict[str, Any]:
    if CURRENT_SESSION_ID is None:
        raise RuntimeError("No active session. Call reset_env() before step_env().")
    return _post("/step", {"session_id": CURRENT_SESSION_ID, "action_type": "raw_command", "params": {"command": action}})


def state_env() -> Dict[str, Any]:
    if CURRENT_SESSION_ID is None:
        raise RuntimeError("No active session. Call reset_env() before state_env().")
    return _get(f"/state?session_id={CURRENT_SESSION_ID}")["state"]


def run_task(task_id: str, seed: int | None = None) -> float:
    effective_seed = BASELINE_SEED if seed is None else seed
    print(f"[START] task={task_id}", flush=True)
    observation = reset_env(task_id, seed=effective_seed)
    action_history: List[Dict[str, Any]] = []
    command_history: List[str] = []
    total_reward = 0.0
    steps = 0

    while steps < MAX_EPISODE_STEPS:
        steps += 1
        state = state_env()
        action = choose_action(observation, state, command_history, adaptive=False)
        error: str | None = None

        try:
            result = step_env(action)
        except Exception as exc:
            error = str(exc)
            print(f"[STEP] step={steps} reward=0.0", flush=True)
            break

        reward = float((result.get("reward") or {}).get("value", 0.0))
        total_reward += reward
        observation = result.get("observation") or observation
        done = bool(result.get("done", False))
        info = result.get("info") or {}
        error = info.get("error")

        command_history.append(action)
        action_history.append(structured_action(action))
        print(f"[STEP] step={steps} reward={reward}", flush=True)

        if done:
            break

    final_state = state_env()
    score = grade_task(task_id, action_history, final_state)
    print(f"[END] task={task_id} score={score} steps={steps}", flush=True)
    return score


def _parse_tasks(argv: List[str]) -> List[str]:
    if len(argv) > 2 and argv[1] == "--task":
        return [argv[2]]
    return TASKS


if __name__ == "__main__":
    for task_name in _parse_tasks(sys.argv):
        run_task(task_name)
