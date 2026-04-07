from typing import Any, Dict, List


def _value(obj: Any):
    return getattr(obj, "value", obj)


def _count_unnecessary_restarts(action_history: List[Any], allowed_target: str) -> int:
    count = 0
    for action in action_history:
        if isinstance(action, dict) and action.get("action") == "restart_service" and action.get("target") != allowed_target:
            count += 1
    return count


def grade(action_history: List[Any], final_state: Dict[str, Any]) -> float:
    services = final_state.get("services", {})
    metrics = final_state.get("metrics", {})
    incident = final_state.get("incident", {})
    sla = final_state.get("sla", {})

    score = 0.0
    if _value(services.get("postgres-primary")) == "healthy":
        score += 0.35
    if _value(services.get("auth-service")) == "healthy" and _value(services.get("billing-service")) == "healthy":
        score += 0.15
    if metrics.get("error_rate", 1.0) <= 0.05 and metrics.get("p99_latency_ms", 9999) <= 600:
        score += 0.2
    if set(["inspect:postgres-primary", "query_metrics", "query_logs", "run_health_check:postgres-primary"]).issubset(set(incident.get("evidence_collected", []))):
        score += 0.1
    if "restart_service:postgres-primary" in incident.get("mitigations_applied", []):
        score += 0.1
    if incident.get("artifact_status", {}).get("runbook_attached"):
        score += 0.03
    if incident.get("artifact_status", {}).get("rca") and incident.get("artifact_status", {}).get("postmortem"):
        score += 0.04
    if sla.get("verified") and sla.get("current_status") == "CURRENTLY_MET":
        score += 0.03

    penalty = 0.0
    penalty += 0.1 * _count_unnecessary_restarts(action_history, "postgres-primary")
    penalty += 0.05 * len(incident.get("unnecessary_restarts", []))
    penalty += 0.03 * incident.get("ignored_alert_ticks", 0)

    return max(0.0, min(1.0, round(score - penalty, 4)))


grade_task_1 = grade
