from typing import Any, Dict, List

from .common import grade_incident


def grade(action_history: List[Any], final_state: Dict[str, Any]) -> float:
    return grade_incident(
        action_history,
        final_state,
        fallback_services=["redis-cache", "cart-service", "frontend-web"],
        fallback_evidence=["inspect:redis-cache", "inspect:cart-service", "query_metrics", "query_logs", "query_traces"],
        fallback_mitigations=["restart_service:redis-cache", "scale:cart-service", "tune_autoscaling:cart-service"],
        error_target=0.055,
        latency_target=700,
        queue_target=5000,
        availability_target=99.1,
    )


grade_task_8 = grade
