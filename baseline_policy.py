import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import requests

from graders.chaos import grade as grade_chaos
from graders.task_1 import grade as grade_task_1
from graders.task_2 import grade as grade_task_2
from graders.task_3 import grade as grade_task_3
from graders.task_4 import grade as grade_task_4
from graders.task_5 import grade as grade_task_5
from graders.task_6 import grade as grade_task_6
from graders.task_7 import grade as grade_task_7
from graders.task_8 import grade as grade_task_8
from graders.task_9 import grade as grade_task_9
from graders.task_10 import grade as grade_task_10
from graders.task_11 import grade as grade_task_11


ENV_URL = os.environ.get("ENV_URL", "https://coolalien35-warroom-deploy.hf.space").rstrip("/")
BASELINE_SEED = int(os.environ.get("BASELINE_SEED", "7"))
REQUEST_TIMEOUT_S = float(os.environ.get("REQUEST_TIMEOUT_S", "30"))
MAX_EPISODE_STEPS = int(os.environ.get("MAX_EPISODE_STEPS", "24"))

TASK_MAPPING = {
    "task_1": "Easy",
    "task_2": "Medium",
    "task_3": "Hard",
    "task_4": "EasyRedis",
    "task_5": "MediumKafka",
    "task_6": "HardMesh",
    "task_7": "MediumReplica",
    "task_8": "MediumCache",
    "task_9": "HardRollback",
    "task_10": "HardDNS",
    "task_11": "HardRegion",
    "chaos": "Chaos",
}

TASKS = ["task_1", "task_2", "task_3", "task_4", "task_5", "task_6", "task_7", "task_8", "task_9", "task_10", "task_11"]

GOOD_VERSIONS = {
    "api-gateway": "v3.2.0",
    "service-mesh": "v1.0.0",
    "frontend-web": "v5.0.4",
    "worker-service": "v3.2.0",
}

ROLE_SRE = "SRE"
ROLE_DEV = "Dev"
ROLE_MANAGER = "Manager"


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"Accept": "application/json", "Content-Type": "application/json"})
    return session


def log_start(task: str, env: str, model: str) -> None:
    print(json.dumps({"type": "START", "task": task, "env": env, "model": model}), flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        json.dumps(
            {
                "type": "STEP",
                "step": int(step),
                "action": action,
                "reward": round(float(reward), 4),
                "done": bool(done),
                "error": error,
            }
        ),
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: float) -> None:
    print(
        json.dumps(
            {
                "type": "END",
                "success": bool(success),
                "steps": int(steps),
                "score": round(float(score), 4),
                "rewards": round(float(rewards), 4),
            }
        ),
        flush=True,
    )


def structured_action(command: str) -> Dict[str, Any]:
    lower = command.lower().strip()
    if lower.startswith("restart service "):
        return {"action": "restart_service", "target": lower.replace("restart service ", "", 1)}
    if lower.startswith("rollback "):
        parts = lower.split()
        return {
            "action": "rollback_deploy",
            "target": parts[1] if len(parts) > 1 else None,
            "version": parts[2] if len(parts) > 2 else None,
        }
    return {"action": lower}


def grade_task(task_id: str, action_history: List[Dict[str, Any]], final_state: Dict[str, Any]) -> float:
    if task_id == "task_1":
        return grade_task_1(action_history, final_state)
    if task_id == "task_2":
        return grade_task_2(action_history, final_state)
    if task_id == "task_3":
        return grade_task_3(action_history, final_state)
    if task_id == "task_4":
        return grade_task_4(action_history, final_state)
    if task_id == "task_5":
        return grade_task_5(action_history, final_state)
    if task_id == "task_6":
        return grade_task_6(action_history, final_state)
    if task_id == "task_7":
        return grade_task_7(action_history, final_state)
    if task_id == "task_8":
        return grade_task_8(action_history, final_state)
    if task_id == "task_9":
        return grade_task_9(action_history, final_state)
    if task_id == "task_10":
        return grade_task_10(action_history, final_state)
    if task_id == "task_11":
        return grade_task_11(action_history, final_state)
    if task_id == "chaos":
        return grade_chaos(action_history, final_state)
    return 0.0


def success(final_state: Dict[str, Any], score: float) -> bool:
    incident = final_state.get("incident") or {}
    metrics = final_state.get("metrics") or {}
    recovered = incident.get("recovery_tick") is not None
    healthy_enough = metrics.get("error_rate", 1.0) <= 0.08 and metrics.get("availability", 0.0) >= 97.0
    return recovered and healthy_enough and score >= 0.7


def completion_score(summary: Dict[str, Any]) -> float:
    artifact_status = summary.get("artifact_status") or {}
    required_artifacts = max(len(artifact_status), 1)
    completed_artifacts = sum(1 for status in artifact_status.values() if status)
    evidence_left = len(summary.get("required_evidence_remaining") or [])
    mitigations_left = len(summary.get("required_mitigations_remaining") or [])
    total_items = required_artifacts + evidence_left + mitigations_left
    completed_items = completed_artifacts + (0 if evidence_left else 1) + (0 if mitigations_left else 1)
    if total_items <= 0:
        return 0.0
    return completed_items / max(total_items, 1)


def command_key(command: str) -> str:
    lower = command.lower().strip()
    patterns = [
        (r"^inspect\s+(.+)$", "inspect:{target}"),
        (r"^query\s+metrics$", "query_metrics"),
        (r"^query\s+logs$", "query_logs"),
        (r"^query\s+traces$", "query_traces"),
        (r"^query\s+topology$", "query_topology"),
        (r"^query\s+deploy(?:\s+history)?$", "query_deploy"),
        (r"^run\s+health\s+check(?:\s+(.+))?$", "run_health_check:{target}"),
        (r"^restart\s+service\s+(.+)$", "restart_service:{target}"),
        (r"^rollback\s+(\S+)\s+(\S+)$", "rollback_deploy:{target}"),
        (r"^clear\s+queue\s+(.+)$", "clear_queue:{target}"),
        (r"^scale\s+(\S+)(?:\s+\d+)?$", "scale:{target}"),
        (r"^tune\s+autoscaling\s+(.+)$", "tune_autoscaling:{target}"),
        (r"^failover\s+zone\s+(.+)$", "failover_zone:{target}"),
        (r"^drain\s+zone\s+(.+)$", "drain_zone:{target}"),
        (r"^restore\s+zone\s+(.+)$", "restore_zone:{target}"),
        (r"^rebalance\s+traffic\s+(.+)$", "rebalance_traffic:{target}"),
        (r"^acknowledge\s+alert.*$", "acknowledge_alert"),
        (r"^notify\s+.*$", "notify"),
        (r"^escalate\s+.*$", "escalate"),
        (r"^update\s+status\s+page.*$", "update_status_page"),
        (r"^attach\s+runbook.*$", "attach_runbook"),
        (r"^verify\s+sla$", "verify_sla"),
        (r"^run\s+rca$", "run_rca"),
        (r"^generate\s+postmortem$", "generate_postmortem"),
    ]
    for pattern, template in patterns:
        match = re.match(pattern, lower)
        if not match:
            continue
        if "{target}" in template:
            target = (match.group(1) if match.groups() else "") or ""
            return template.format(target=target)
        return template
    return lower.replace(" ", "_")


def role_for_command(command: str) -> str:
    lower = command.lower()
    if lower.startswith(("attach runbook", "generate postmortem", "query deploy", "rollback ")):
        return ROLE_DEV
    if lower.startswith(("notify ", "update status page", "escalate ")):
        return ROLE_MANAGER
    if lower.startswith("verify sla"):
        return ROLE_SRE
    if lower.startswith("switch role "):
        target = command.split(" ", 2)[-1].strip()
        return ROLE_DEV if target.lower() == "dev" else ROLE_MANAGER if target.lower() == "manager" else ROLE_SRE
    return ROLE_SRE


def _missing_communications(state: Dict[str, Any]) -> List[str]:
    incident = state.get("incident") or {}
    if not incident.get("needs_manager_comms"):
        return []
    pending: List[str] = []
    if not incident.get("notified"):
        pending.append("notify stakeholders")
    if not incident.get("status_page_updated"):
        pending.append("update status page")
    if not incident.get("escalated_to"):
        pending.append("escalate leadership")
    return pending


def _generic_rollback_target(state: Dict[str, Any], summary: Dict[str, Any], services: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    variant = summary.get("variant") or {}
    deploy_target = variant.get("deploy_target")
    if deploy_target:
        return deploy_target, GOOD_VERSIONS.get(deploy_target, "v3.2.0")
    deploy_targets = (state.get("incident") or {}).get("deploy_targets") or {}
    if deploy_targets:
        service, version = sorted(deploy_targets.items())[0]
        return service, version
    for service in sorted(services):
        if services.get(service) in {"down", "degraded"} and service in GOOD_VERSIONS:
            return service, GOOD_VERSIONS[service]
    return None


def _service_candidates(summary: Dict[str, Any], services: Dict[str, Any]) -> List[str]:
    targets = list(summary.get("service_targets") or [])
    degraded = [name for name, status in sorted(services.items()) if status in {"down", "degraded", "restarting", "isolated"}]
    ordered: List[str] = []
    for name in targets + degraded:
        if name not in ordered:
            ordered.append(name)
    return ordered


def _zone_candidates(summary: Dict[str, Any], zones: Dict[str, Any]) -> List[str]:
    targets = list(summary.get("zone_targets") or [])
    degraded = [zone for zone, status in sorted(zones.items()) if status in {"down", "degraded"}]
    ordered: List[str] = []
    for name in targets + degraded:
        if name not in ordered:
            ordered.append(name)
    return ordered


def _candidate_commands(observation: Dict[str, Any], state: Dict[str, Any], history: List[str], adaptive: bool) -> List[str]:
    summary = observation.get("incident_summary") or {}
    metrics = observation.get("metrics") or {}
    services = {name: str(status) for name, status in (observation.get("services") or {}).items()}
    zones = {name: str(status) for name, status in (observation.get("zone_health") or {}).items()}
    suggestions = observation.get("suggested_actions") or []
    incident = state.get("incident") or {}
    artifact_status = summary.get("artifact_status") or {}
    candidates: List[str] = []

    if observation.get("alerts") and "acknowledge alert" not in history:
        candidates.append("acknowledge alert")

    for item in summary.get("required_evidence_remaining") or []:
        if item.startswith("inspect:"):
            candidates.append(f"inspect {item.split(':', 1)[1]}")
        elif item == "query_metrics":
            candidates.append("query metrics")
        elif item == "query_logs":
            candidates.append("query logs")
        elif item == "query_traces":
            candidates.append("query traces")
        elif item == "query_topology":
            candidates.append("query topology")
        elif item == "query_deploy":
            candidates.append("query deploy history")
        elif item.startswith("run_health_check:"):
            candidates.append(f"run health check {item.split(':', 1)[1]}")

    for item in summary.get("required_mitigations_remaining") or []:
        action_name, _, target = item.partition(":")
        if action_name == "restart_service" and target:
            candidates.append(f"restart service {target}")
        elif action_name == "rollback_deploy":
            rollback = _generic_rollback_target(state, summary, services)
            if rollback:
                candidates.append(f"rollback {rollback[0]} {rollback[1]}")
        elif action_name == "clear_queue" and target:
            candidates.append(f"clear queue {target}")
        elif action_name == "scale" and target:
            candidates.append(f"scale {target} 3")
        elif action_name == "tune_autoscaling" and target:
            candidates.append(f"tune autoscaling {target}")
        elif action_name == "failover_zone" and target:
            candidates.append(f"failover zone {target}")
        elif action_name == "drain_zone" and target:
            candidates.append(f"drain zone {target}")
        elif action_name == "restore_zone" and target:
            candidates.append(f"restore zone {target}")
        elif action_name == "rebalance_traffic" and target:
            candidates.append(f"rebalance traffic {target}")

    for command in _missing_communications(state):
        candidates.append(command)

    if not artifact_status.get("sla_verified"):
        candidates.append("verify sla")
    if not artifact_status.get("rca"):
        candidates.append("run rca")
    if not artifact_status.get("runbook_attached"):
        candidates.append("attach runbook")
    if artifact_status.get("rca") and not artifact_status.get("postmortem"):
        candidates.append("generate postmortem")

    for command in suggestions:
        candidates.append(command)

    queue_depth = float(metrics.get("queue_depth", 0) or 0)
    if queue_depth > 2000:
        for service in _service_candidates(summary, services):
            if "worker" in service or "kafka" in service or "recommendation" in service:
                candidates.extend([f"inspect {service}", f"clear queue {service}", f"scale {service} 3", f"tune autoscaling {service}"])

    rollback = _generic_rollback_target(state, summary, services)
    if rollback:
        candidates.extend([f"inspect {rollback[0]}", "query deploy history", f"rollback {rollback[0]} {rollback[1]}"])

    for service in _service_candidates(summary, services):
        candidates.extend([f"inspect {service}", f"run health check {service}", f"restart service {service}"])

    for zone in _zone_candidates(summary, zones):
        candidates.extend([f"failover zone {zone}", f"drain zone {zone}", f"restore zone {zone}"])

    if adaptive:
        if metrics.get("error_rate", 0.0) > 0.08:
            candidates.append("query metrics")
        if metrics.get("p99_latency_ms", 0) > 450:
            candidates.append("query traces")
        if any(status in {"down", "degraded"} for status in services.values()):
            candidates.append("query logs")
        if any(status in {"down", "degraded"} for status in zones.values()):
            candidates.append("query topology")

    candidates.extend(["query metrics", "query logs", "query traces", "query topology"])

    deduped: List[str] = []
    for command in candidates:
        if command not in deduped and command not in history:
            deduped.append(command)
    return deduped


def _command_score(command: str, observation: Dict[str, Any], state: Dict[str, Any], history: List[str], adaptive: bool) -> float:
    summary = observation.get("incident_summary") or {}
    metrics = observation.get("metrics") or {}
    services = {name: str(status) for name, status in (observation.get("services") or {}).items()}
    zones = {name: str(status) for name, status in (observation.get("zone_health") or {}).items()}
    incident = state.get("incident") or {}
    required_evidence = set(summary.get("required_evidence_remaining") or [])
    required_mitigations = set(summary.get("required_mitigations_remaining") or [])
    artifact_status = summary.get("artifact_status") or {}
    score = 0.0
    key = command_key(command)
    lower = command.lower()

    if key in required_evidence:
        score += 100.0
    if key in required_mitigations:
        score += 120.0
    if command in (observation.get("suggested_actions") or []):
        score += 15.0

    if lower.startswith("acknowledge alert") and observation.get("alerts"):
        score += 90.0

    if lower.startswith("query metrics"):
        score += 20.0 if metrics.get("error_rate", 0.0) > 0.04 or metrics.get("availability", 100.0) < 99.0 else 5.0
    if lower.startswith("query logs"):
        score += 18.0 if any(status in {"down", "degraded"} for status in services.values()) else 4.0
    if lower.startswith("query traces"):
        score += 18.0 if metrics.get("p99_latency_ms", 0) > 400 else 3.0
    if lower.startswith("query topology"):
        score += 20.0 if any(status in {"down", "degraded"} for status in zones.values()) else 4.0
    if lower.startswith("query deploy"):
        score += 26.0 if _generic_rollback_target(state, summary, services) else 0.0

    if lower.startswith("inspect "):
        target = lower.replace("inspect ", "", 1)
        status = services.get(target)
        if status == "down":
            score += 60.0
        elif status in {"degraded", "restarting", "isolated"}:
            score += 40.0
        if target in (summary.get("service_targets") or []):
            score += 20.0

    if lower.startswith("run health check "):
        target = lower.replace("run health check ", "", 1)
        if services.get(target) in {"down", "degraded", "restarting"}:
            score += 34.0

    if lower.startswith("restart service "):
        target = lower.replace("restart service ", "", 1)
        status = services.get(target)
        if status == "down":
            score += 80.0
        elif status in {"degraded", "restarting"}:
            score += 45.0

    if lower.startswith("clear queue "):
        target = lower.replace("clear queue ", "", 1)
        score += min(70.0, float(metrics.get("queue_depth", 0) or 0) / 120.0)
        if "worker" in target or "kafka" in target:
            score += 12.0

    if lower.startswith("scale ") or lower.startswith("tune autoscaling "):
        score += min(55.0, float(metrics.get("queue_depth", 0) or 0) / 180.0)

    if lower.startswith("rollback "):
        rollback = _generic_rollback_target(state, summary, services)
        if rollback and rollback[0] in lower:
            score += 95.0

    if lower.startswith(("failover zone ", "drain zone ", "restore zone ")):
        zone = lower.split()[-1]
        zone_status = zones.get(zone)
        if lower.startswith("restore zone "):
            score += 60.0 if zone_status in {"down", "degraded"} else 8.0
        elif lower.startswith("failover zone "):
            score += 75.0 if zone_status in {"down", "degraded"} else 5.0
        else:
            score += 54.0 if zone_status in {"down", "degraded"} else 6.0

    if lower.startswith("rebalance traffic "):
        target = lower.replace("rebalance traffic ", "", 1)
        if services.get(target) in {"down", "degraded"} or target in (summary.get("service_targets") or []):
            score += 50.0

    if lower.startswith("notify "):
        score += 38.0 if incident.get("needs_manager_comms") else 4.0
    if lower.startswith("update status page"):
        score += 42.0 if incident.get("needs_manager_comms") else 4.0
    if lower.startswith("escalate "):
        score += 34.0 if incident.get("needs_manager_comms") else 4.0

    if lower.startswith("verify sla"):
        score += 48.0 if completion_score(summary) >= 0.5 else 8.0
    if lower.startswith("run rca"):
        score += 44.0 if len(required_evidence) <= 1 else 20.0
        if artifact_status.get("rca"):
            score -= 30.0
    if lower.startswith("attach runbook"):
        score += 40.0 if not artifact_status.get("runbook_attached") else -20.0
    if lower.startswith("generate postmortem"):
        score += 52.0 if artifact_status.get("rca") else 10.0
        if artifact_status.get("postmortem"):
            score -= 30.0

    repetitions = history.count(command)
    score -= repetitions * 25.0

    if adaptive:
        score += 4.0 if lower.startswith(("query metrics", "query logs", "query traces", "query topology")) else 0.0

    return score


def choose_action(observation: Dict[str, Any], state: Dict[str, Any], history: List[str], adaptive: bool = False) -> str:
    current_role = observation.get("current_role", ROLE_SRE)
    candidates = _candidate_commands(observation, state, history, adaptive=adaptive)
    if not candidates:
        return "query metrics" if current_role == ROLE_SRE else f"switch role {ROLE_SRE}"

    ranked = sorted(
        ((-_command_score(command, observation, state, history, adaptive), command) for command in candidates),
        key=lambda item: (item[0], item[1]),
    )
    chosen = ranked[0][1]
    needed_role = role_for_command(chosen)
    if current_role != needed_role:
        return f"switch role {needed_role}"
    return chosen
