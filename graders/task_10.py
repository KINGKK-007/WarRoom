from typing import Any, Dict, List

from .common import grade_incident


def grade(action_history: List[Any], final_state: Dict[str, Any]) -> float:
    return grade_incident(
        action_history,
        final_state,
        fallback_services=["dns-control-plane", "edge-proxy", "api-gateway"],
        fallback_zones=["us-east-1a"],
        fallback_evidence=["inspect:dns-control-plane", "inspect:edge-proxy", "query_metrics", "query_logs", "query_topology"],
        fallback_mitigations=["restart_service:dns-control-plane", "failover_zone:us-east-1a", "rebalance_traffic:api-gateway"],
        error_target=0.05,
        latency_target=650,
        availability_target=99.0,
        ignored_alert_weight=0.04,
    )


grade_task_10 = grade
