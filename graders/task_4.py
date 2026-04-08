from typing import Any, Dict, List

from .common import grade_incident


def grade(action_history: List[Any], final_state: Dict[str, Any]) -> float:
    return grade_incident(
        action_history,
        final_state,
        fallback_services=["redis-session", "auth-service", "cart-service"],
        fallback_evidence=["inspect:redis-session", "query_metrics", "query_logs", "run_health_check:redis-session"],
        fallback_mitigations=["restart_service:redis-session"],
        error_target=0.05,
        latency_target=600,
        availability_target=99.0,
        ignored_alert_weight=0.03,
    )


grade_task_4 = grade
