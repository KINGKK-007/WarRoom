from typing import Any, Dict, List

from .common import grade_incident


def grade(action_history: List[Any], final_state: Dict[str, Any]) -> float:
    incident = final_state.get("incident", {})
    deploy_target = next(iter((incident.get("deploy_targets") or {"api-gateway": "v3.2.0"}).keys()))
    inspect_target = "frontend-web" if deploy_target == "frontend-web" else "api-gateway"
    fallback_services = [deploy_target, "frontend-web", "mobile-bff"]
    fallback_mitigations = [f"rollback_deploy:{deploy_target}", "failover_zone:us-east-1c", f"rebalance_traffic:{deploy_target}"]

    return grade_incident(
        action_history,
        final_state,
        fallback_services=fallback_services,
        fallback_zones=["us-east-1c"],
        fallback_evidence=["query_metrics", "query_traces", "query_deploy", "query_topology", f"inspect:{inspect_target}"],
        fallback_mitigations=fallback_mitigations,
        error_target=0.05,
        latency_target=600,
        availability_target=99.0,
        ignored_alert_weight=0.04,
    )


grade_task_3 = grade
