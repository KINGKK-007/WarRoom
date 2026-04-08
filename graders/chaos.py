from typing import Any, Dict, List

from .common import grade_incident


def grade(action_history: List[Any], final_state: Dict[str, Any]) -> float:
    return grade_incident(
        action_history,
        final_state,
        error_target=0.06,
        latency_target=700,
        queue_target=8000,
        availability_target=98.5,
        ignored_alert_weight=0.05,
        service_weight=0.28,
        zone_weight=0.12,
        metric_weight=0.16,
        evidence_weight=0.14,
        mitigation_weight=0.14,
        artifact_weight=0.08,
        communication_weight=0.05,
        sla_weight=0.03,
    )


grade_chaos = grade
