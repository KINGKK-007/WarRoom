from typing import Any, Dict, List

from .common import grade_incident


def grade(action_history: List[Any], final_state: Dict[str, Any]) -> float:
    return grade_incident(
        action_history,
        final_state,
        fallback_services=["kafka", "worker-service", "notification-service", "analytics-service"],
        fallback_zones=["us-central-1a"],
        fallback_evidence=["inspect:kafka", "inspect:worker-service", "query_metrics", "query_logs", "query_traces", "query_topology"],
        fallback_mitigations=["restart_service:kafka", "clear_queue:worker-service", "scale:worker-service", "drain_zone:us-central-1a", "restore_zone:us-central-1a"],
        error_target=0.05,
        latency_target=650,
        queue_target=5000,
        availability_target=99.0,
        ignored_alert_weight=0.04,
    )


grade_task_5 = grade
