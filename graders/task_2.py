from typing import Any, Dict, List

from .common import grade_incident


def grade(action_history: List[Any], final_state: Dict[str, Any]) -> float:
    return grade_incident(
        action_history,
        final_state,
        fallback_services=["worker-service", "scheduler-service", "notification-service"],
        fallback_zones=["us-east-1b"],
        fallback_evidence=["inspect:worker-service", "query_metrics", "query_logs", "query_traces", "query_topology"],
        fallback_mitigations=["clear_queue:worker-service", "scale:worker-service", "tune_autoscaling:worker-service", "drain_zone:us-east-1b", "restore_zone:us-east-1b"],
        error_target=0.05,
        latency_target=600,
        queue_target=5000,
        availability_target=99.0,
        ignored_alert_weight=0.05,
    )
