from typing import Any, Dict, List

from .common import grade_incident


def grade(action_history: List[Any], final_state: Dict[str, Any]) -> float:
    return grade_incident(
        action_history,
        final_state,
        fallback_services=["frontend-web", "api-gateway"],
        fallback_evidence=["inspect:frontend-web", "query_metrics", "query_logs", "query_traces", "query_deploy"],
        fallback_mitigations=["rollback_deploy:frontend-web", "restart_service:frontend-web", "rebalance_traffic:api-gateway"],
        error_target=0.05,
        latency_target=650,
        availability_target=99.0,
        ignored_alert_weight=0.04,
    )


grade_task_9 = grade
