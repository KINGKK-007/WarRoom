"""
adaptive_inference.py — heuristic adaptive baseline for DevOps War Room

Unlike the deterministic baseline in inference.py, this agent branches on
live observation and can handle task variants and the Chaos scenario.
"""

import json
import os
import sys
import time
from typing import Any, Dict, List

import requests

from graders.task_1 import grade as grade_task_1
from graders.task_2 import grade as grade_task_2
from graders.task_3 import grade as grade_task_3


ENV_URL = os.environ.get("ENV_URL", "https://coolalien35-warroom-deploy.hf.space")
TASK_MAPPING = {"task_1": "Easy", "task_2": "Medium", "task_3": "Hard", "chaos": "Chaos"}


def reset_env(task_id: str, seed: int | None = None):
    scenario = TASK_MAPPING.get(task_id, task_id)
    payload: Dict[str, Any] = {"task_id": scenario}
    if seed is not None:
        payload["seed"] = seed
    response = requests.post(f"{ENV_URL}/reset", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def step_env(action: str):
    response = requests.post(
        f"{ENV_URL}/step",
        json={"action_type": "raw_command", "params": {"command": action}},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def state_env():
    response = requests.get(f"{ENV_URL}/state", timeout=30)
    response.raise_for_status()
    return response.json()


def _structured_action(command: str):
    lower = command.lower().strip()
    if lower.startswith("restart service "):
        return {"action": "restart_service", "target": lower.replace("restart service ", "", 1)}
    if lower.startswith("rollback "):
        parts = lower.split()
        return {"action": "rollback_deploy", "target": parts[1], "version": parts[2]}
    return {"action": lower}


def _pick_unfinished(prefixes: List[str], completed: List[str]) -> str | None:
    for command in prefixes:
        if command not in completed:
            return command
    return None


def _degraded_zones(zones: Dict[str, Any]) -> List[str]:
    return [zone for zone, status in zones.items() if status in {"down", "degraded"}]


def _choose_recovery_flow(observation: Dict[str, Any]) -> List[str]:
    metrics = observation.get("metrics") or {}
    services = observation.get("services", {})
    zones = observation.get("zone_health") or {}
    summary = observation.get("incident_summary") or {}
    variant = summary.get("variant") or {}
    incident_name = summary.get("name", "")
    degraded_zones = _degraded_zones(zones)

    if services.get("postgres-primary") == "down" or incident_name == "primary-database-outage":
        return [
            "acknowledge alert",
            "inspect postgres-primary",
            "query metrics",
            "query logs",
            "run health check postgres-primary",
            "restart service postgres-primary",
            "verify sla",
            "run rca",
            "switch role Dev",
            "attach runbook",
            "generate postmortem",
        ]

    if metrics.get("queue_depth", 0) > 5000 or incident_name == "queue-backlog-and-zone-degradation":
        focus_zone = variant.get("focus_zone")
        if not focus_zone:
            focus_zone = degraded_zones[0] if degraded_zones else "us-east-1b"
        return [
            "acknowledge alert",
            "inspect worker-service",
            "query metrics",
            "query logs",
            "query traces",
            "query topology",
            "clear queue worker-service",
            "scale worker-service 3",
            "tune autoscaling worker-service",
            f"drain zone {focus_zone}",
            f"restore zone {focus_zone}",
            "verify sla",
            "run rca",
            "switch role Dev",
            "attach runbook",
            "generate postmortem",
        ]

    deploy_target = variant.get("deploy_target", "api-gateway")
    rollback_version = "v3.2.0" if deploy_target == "api-gateway" else "v5.0.4"
    focus_zone = variant.get("focus_zone")
    if not focus_zone:
        focus_zone = degraded_zones[0] if degraded_zones else "us-east-1c"
    if (
        metrics.get("p99_latency_ms", 0) > 800
        or services.get("api-gateway") == "degraded"
        or incident_name == "bad-gateway-deploy-with-zone-failure"
    ):
        return [
            "acknowledge alert",
            "query metrics",
            "query traces",
            "query topology",
            f"inspect {deploy_target}",
            "switch role Dev",
            "query deploy history",
            f"rollback {deploy_target} {rollback_version}",
            "attach runbook",
            "switch role SRE",
            f"failover zone {focus_zone}",
            f"rebalance traffic {deploy_target}",
            f"restore zone {focus_zone}",
            "verify sla",
            "run rca",
            "switch role Dev",
            "generate postmortem",
        ]

    primary_down = services.get("postgres-primary") == "down"
    flow: List[str] = ["acknowledge alert", "query metrics", "query traces", "query topology"]
    if primary_down:
        flow.extend(["inspect postgres-primary", "restart service postgres-primary"])
    if services.get("worker-service") in {"down", "degraded"} and metrics.get("queue_depth", 0) > 5000:
        flow.extend(["inspect worker-service", "clear queue worker-service", "scale worker-service 3", "tune autoscaling worker-service"])
    for zone in degraded_zones:
        flow.append(f"failover zone {zone}")
    if services.get("api-gateway") in {"down", "degraded"}:
        flow.extend(["switch role Dev", "query deploy history", "rollback api-gateway v3.2.0", "attach runbook", "switch role SRE", "rebalance traffic api-gateway"])
    for zone in degraded_zones:
        flow.append(f"restore zone {zone}")
    flow.extend(["verify sla", "run rca", "switch role Dev", "generate postmortem"])
    return flow


def choose_action(observation: Dict[str, Any], history: List[str]) -> str:
    current_role = observation.get("current_role", "SRE")
    plan = _choose_recovery_flow(observation)

    for command in plan:
        if command in history:
            continue
        lower = command.lower()
        if lower.startswith("switch role "):
            target_role = lower.replace("switch role ", "", 1)
            if current_role.lower() == target_role:
                continue
        if lower.startswith("attach runbook") or lower.startswith("generate postmortem") or lower.startswith("query deploy") or lower.startswith("rollback "):
            if current_role != "Dev":
                return "switch role Dev"
        if lower.startswith("verify sla"):
            if current_role not in {"SRE", "Manager"}:
                return "switch role SRE"
        if lower.startswith("failover zone") or lower.startswith("restore zone") or lower.startswith("rebalance traffic") or lower.startswith("clear queue") or lower.startswith("scale ") or lower.startswith("tune autoscaling") or lower.startswith("restart service") or lower.startswith("run health check") or lower.startswith("inspect ") or lower.startswith("query metrics") or lower.startswith("query topology"):
            if current_role != "SRE":
                return "switch role SRE"
        return command

    if current_role != "SRE":
        return "switch role SRE"
    return "query metrics"


def _grade(task_id: str, actions: List[Dict[str, Any]], state: Dict[str, Any]) -> float:
    if task_id == "task_1":
        return grade_task_1(actions, state)
    if task_id == "task_2":
        return grade_task_2(actions, state)
    if task_id == "task_3":
        return grade_task_3(actions, state)
    return 0.0


def run_task(task_id: str, seed: int | None = None):
    print(json.dumps({"type": "START", "task": task_id, "timestamp": time.time()}))
    obs = reset_env(task_id, seed=seed)
    commands: List[str] = []
    actions: List[Dict[str, Any]] = []
    total_reward = 0.0
    step_num = 0

    while step_num < 24:
        step_num += 1
        action = choose_action(obs, commands)
        result = step_env(action)
        reward = result["reward"]["value"]
        obs = result["observation"]
        commands.append(action)
        actions.append(_structured_action(action))
        total_reward += reward
        print(json.dumps({"type": "STEP", "step": step_num, "action": action, "reward": round(reward, 4), "done": result["done"], "error": None}))
        if result["done"]:
            state = state_env()
            if state.get("incident", {}).get("recovery_tick") is not None:
                break

    final_state = state_env()
    score = _grade(task_id, actions, final_state)
    print(json.dumps({"type": "END", "task": task_id, "total_reward": round(total_reward, 4), "steps": step_num, "timestamp": time.time()}))
    print(json.dumps({"type": "SCORE", "task": task_id, "score": round(score, 4)}))
    return score


if __name__ == "__main__":
    task_arg = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "--task" else None
    tasks = [task_arg] if task_arg else ["task_1", "task_2", "task_3"]
    for task in tasks:
        run_task(task)
