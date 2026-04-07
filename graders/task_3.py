from typing import Any, Dict, List


def _value(obj: Any):
    return getattr(obj, "value", obj)


def _unnecessary_restart_penalty(action_history: List[Any]) -> float:
    penalty = 0.0
    for action in action_history:
        if isinstance(action, dict) and action.get("action") == "restart_service":
            penalty += 0.08
    return penalty


def grade(action_history: List[Any], final_state: Dict[str, Any]) -> float:
    services = final_state.get("services", {})
    metrics = final_state.get("metrics", {})
    incident = final_state.get("incident", {})
    zones = final_state.get("zones", {})
    sla = final_state.get("sla", {})

    score = 0.0
    if _value(services.get("api-gateway")) == "healthy":
        score += 0.18
    if _value(services.get("frontend-web")) == "healthy" and _value(services.get("mobile-bff")) == "healthy":
        score += 0.12
    if zones.get("us-east-1c", {}).get("status") == "healthy":
        score += 0.12
    if metrics.get("p99_latency_ms", 9999) <= 600 and metrics.get("error_rate", 1.0) <= 0.05:
        score += 0.18
    if set(["query_metrics", "query_traces", "query_deploy", "query_topology", "inspect:api-gateway"]).issubset(set(incident.get("evidence_collected", []))):
        score += 0.08
    if set(["rollback_deploy:api-gateway", "failover_zone:us-east-1c", "rebalance_traffic:api-gateway"]).issubset(set(incident.get("mitigations_applied", []))):
        score += 0.14
    if incident.get("artifact_status", {}).get("runbook_attached"):
        score += 0.04
    if incident.get("artifact_status", {}).get("rca") and incident.get("artifact_status", {}).get("postmortem"):
        score += 0.08
    if sla.get("verified") and sla.get("current_status") == "CURRENTLY_MET":
        score += 0.06

    penalty = _unnecessary_restart_penalty(action_history)
    penalty += 0.04 * incident.get("ignored_alert_ticks", 0)
    return max(0.0, min(1.0, round(score - penalty, 4)))


grade_task_3 = grade
