from typing import Any, Dict, List


def _value(obj: Any):
    return getattr(obj, "value", obj)


def _restart_penalty(action_history: List[Any], healthy_target: str) -> float:
    penalty = 0.0
    for action in action_history:
        if isinstance(action, dict) and action.get("action") == "restart_service" and action.get("target") != healthy_target:
            penalty += 0.08
    return penalty


def grade(action_history: List[Any], final_state: Dict[str, Any]) -> float:
    services = final_state.get("services", {})
    metrics = final_state.get("metrics", {})
    incident = final_state.get("incident", {})
    zones = final_state.get("zones", {})

    score = 0.0
    if _value(services.get("worker-service")) == "healthy":
        score += 0.22
    if _value(services.get("scheduler-service")) == "healthy" and _value(services.get("notification-service")) == "healthy":
        score += 0.14
    if zones.get("us-east-1b", {}).get("status") == "healthy":
        score += 0.1
    if metrics.get("queue_depth", 999999) <= 5000:
        score += 0.18
    if metrics.get("error_rate", 1.0) <= 0.05 and metrics.get("p99_latency_ms", 9999) <= 600:
        score += 0.1
    if set(["inspect:worker-service", "query_metrics", "query_logs", "query_traces", "query_topology"]).issubset(set(incident.get("evidence_collected", []))):
        score += 0.08
    if set(["clear_queue:worker-service", "scale:worker-service", "tune_autoscaling:worker-service", "drain_zone:us-east-1b", "restore_zone:us-east-1b"]).issubset(set(incident.get("mitigations_applied", []))):
        score += 0.1
    if incident.get("artifact_status", {}).get("rca") and incident.get("artifact_status", {}).get("postmortem"):
        score += 0.04
    if incident.get("artifact_status", {}).get("runbook_attached"):
        score += 0.04

    penalty = _restart_penalty(action_history, "worker-service")
    penalty += 0.05 * incident.get("ignored_alert_ticks", 0)
    return max(0.0, min(1.0, round(score - penalty, 4)))
