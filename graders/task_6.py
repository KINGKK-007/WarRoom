from typing import Any, Dict, List

from .common import grade_incident


def grade(action_history: List[Any], final_state: Dict[str, Any]) -> float:
    return grade_incident(
        action_history,
        final_state,
        fallback_services=["service-mesh", "api-gateway", "frontend-web"],
        fallback_zones=["us-east-1a"],
        fallback_evidence=["query_metrics", "query_traces", "query_deploy", "query_topology", "inspect:service-mesh"],
        fallback_mitigations=["rollback_deploy:service-mesh", "failover_zone:us-east-1a", "rebalance_traffic:api-gateway"],
        error_target=0.05,
        latency_target=650,
        availability_target=99.0,
        ignored_alert_weight=0.04,
    )


grade_task_6 = grade
