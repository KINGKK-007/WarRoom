from typing import Any, Dict, List

from .common import grade_incident


def grade(action_history: List[Any], final_state: Dict[str, Any]) -> float:
    return grade_incident(
        action_history,
        final_state,
        fallback_services=["postgres-replica", "profile-service", "report-service"],
        fallback_zones=["us-east-1b"],
        fallback_evidence=["inspect:postgres-replica", "query_metrics", "query_logs", "query_traces", "run_health_check:postgres-replica"],
        fallback_mitigations=["restart_service:postgres-replica", "restore_zone:us-east-1b"],
        error_target=0.05,
        latency_target=650,
        queue_target=4000,
        availability_target=99.3,
    )


grade_task_7 = grade
