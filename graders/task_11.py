from typing import Any, Dict, List

from .common import grade_incident


def grade(action_history: List[Any], final_state: Dict[str, Any]) -> float:
    return grade_incident(
        action_history,
        final_state,
        fallback_services=["api-gateway", "frontend-web", "order-service", "payment-service"],
        fallback_zones=["us-east-1a", "us-east-1b"],
        fallback_evidence=["inspect:api-gateway", "query_metrics", "query_logs", "query_traces", "query_topology"],
        fallback_mitigations=["failover_zone:us-east-1a", "failover_zone:us-east-1b", "restore_zone:us-east-1a", "restore_zone:us-east-1b", "rebalance_traffic:api-gateway"],
        error_target=0.05,
        latency_target=700,
        availability_target=99.0,
        ignored_alert_weight=0.05,
    )


grade_task_11 = grade
