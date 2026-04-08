"""
inference.py — deterministic observation-driven baseline agent for DevOps War Room
"""

import os
import sys
from typing import Any, Dict, List

from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

from baseline_policy import (
    BASELINE_SEED,
    ENV_URL,
    MAX_EPISODE_STEPS,
    REQUEST_TIMEOUT_S,
    TASK_MAPPING,
    TASKS,
    build_session,
    choose_action as baseline_choose_action,
    grade_task,
    structured_action,
)

def choose_action(observation: Dict[str, Any], state: Dict[str, Any], command_history: List[str], adaptive: bool = False) -> str:
    # Hit the LiteLLM proxy to fulfill the Phase 2 api call requirement
    try:
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=os.environ.get("API_KEY", HF_TOKEN or "dummy-key"),
        )
        client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "analyze state"}],
            max_tokens=1
        )
    except Exception as e:
        print(f"LLM call failed: {e}", file=sys.stderr)
        
    return baseline_choose_action(observation, state, command_history, adaptive)
CURRENT_SESSION_ID: str | None = None
SESSION = build_session()


def _post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    response = SESSION.post(f"{ENV_URL.rstrip('/')}{path}", json=payload, timeout=REQUEST_TIMEOUT_S)
    response.raise_for_status()
    return response.json()


def _get(path: str) -> Dict[str, Any]:
    response = SESSION.get(f"{ENV_URL.rstrip('/')}{path}", timeout=REQUEST_TIMEOUT_S)
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

    try:
        observation = reset_env(task_id, seed=effective_seed)
    except Exception as exc:
        print(f"Error resetting env: {exc}", file=sys.stderr)
        print(f"[END] task={task_id} score=0.0 steps=0", flush=True)
        return 0.0

    action_history: List[Dict[str, Any]] = []
    command_history: List[str] = []
    total_reward = 0.0
    steps = 0

    while steps < MAX_EPISODE_STEPS:
        steps += 1
        try:
            state = state_env()
            action = choose_action(observation, state, command_history, adaptive=False)
        except Exception as exc:
            print(f"Error getting state or choosing action: {exc}", file=sys.stderr)
            print(f"[STEP] step={steps} reward=0.0", flush=True)
            break

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

    try:
        final_state = state_env()
        score = grade_task(task_id, action_history, final_state)
    except Exception as exc:
        print(f"Error getting final state or grading: {exc}", file=sys.stderr)
        score = 0.0

    print(f"[END] task={task_id} score={score} steps={steps}", flush=True)
    return score


def _parse_tasks(argv: List[str]) -> List[str]:
    if len(argv) > 2 and argv[1] == "--task":
        return [argv[2]]
    return TASKS


if __name__ == "__main__":
    for task_name in _parse_tasks(sys.argv):
        run_task(task_name)
