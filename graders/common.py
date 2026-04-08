from typing import Any, Dict, Iterable, List, Sequence


def value(obj: Any):
    return getattr(obj, "value", obj)


def clamp_score(score: float) -> float:
    return max(0.0, min(1.0, round(score, 4)))


def ratio_hits(values: Iterable[bool]) -> float:
    values = list(values)
    if not values:
        return 1.0
    return sum(1.0 for item in values if item) / len(values)


def service_health_ratio(services: Dict[str, Any], targets: Sequence[str]) -> float:
    return ratio_hits(value(services.get(target)) == "healthy" for target in targets)


def zone_health_ratio(zones: Dict[str, Any], targets: Sequence[str]) -> float:
    return ratio_hits((zones.get(target) or {}).get("status") == "healthy" for target in targets)


def evidence_ratio(incident: Dict[str, Any], required: Sequence[str]) -> float:
    collected = set(incident.get("evidence_collected", []))
    return ratio_hits(item in collected for item in required)


def mitigation_ratio(incident: Dict[str, Any], required: Sequence[str]) -> float:
    mitigations = set(incident.get("mitigations_applied", []))
    return ratio_hits(item in mitigations for item in required)


def artifact_ratio(incident: Dict[str, Any], keys: Sequence[str]) -> float:
    status = incident.get("artifact_status", {})
    return ratio_hits(bool(status.get(key)) for key in keys)


def communication_ratio(incident: Dict[str, Any], action_history: List[Any]) -> float:
    if not incident.get("needs_manager_comms"):
        return 1.0

    executed = set(incident.get("production_actions_completed", []))
    for action in action_history:
        if not isinstance(action, dict):
            continue
        action_name = action.get("action")
        if action_name in {"notify", "escalate", "update_status_page"}:
            executed.add(action_name)

    checks = [
        len(incident.get("notified", [])) > 0 or "notify" in executed,
        len(incident.get("escalated_to", [])) > 0 or "escalate" in executed,
        bool(incident.get("status_page_updated")) or "update_status_page" in executed,
    ]
    return ratio_hits(checks)


def metric_ratio(
    metrics: Dict[str, Any],
    *,
    error_target: float,
    latency_target: int,
    queue_target: int | None = None,
    availability_target: float | None = None,
) -> float:
    checks: List[float] = []

    error_rate = metrics.get("error_rate")
    if error_rate is not None:
        checks.append(1.0 if error_rate <= error_target else max(0.0, error_target / max(error_rate, 1e-6)))

    latency = metrics.get("p99_latency_ms")
    if latency is not None:
        checks.append(1.0 if latency <= latency_target else max(0.0, latency_target / max(latency, 1)))

    if queue_target is not None and "queue_depth" in metrics:
        queue_depth = metrics.get("queue_depth", queue_target)
        checks.append(1.0 if queue_depth <= queue_target else max(0.0, queue_target / max(queue_depth, 1)))

    if availability_target is not None and "availability" in metrics:
        availability = metrics.get("availability", availability_target)
        checks.append(min(1.0, max(0.0, availability / availability_target)))

    if not checks:
        return 0.0
    return sum(checks) / len(checks)


def _preferred(items: Sequence[str], fallback: Sequence[str]) -> List[str]:
    values = [item for item in items if item]
    return values or list(fallback)


def dynamic_targets(incident: Dict[str, Any], *, fallback_services: Sequence[str] = (), fallback_zones: Sequence[str] = ()) -> tuple[List[str], List[str]]:
    services = _preferred(incident.get("service_targets", []), fallback_services)
    zones = _preferred(incident.get("zone_targets", []), fallback_zones)
    return services, zones


def dynamic_requirements(
    incident: Dict[str, Any],
    *,
    fallback_evidence: Sequence[str] = (),
    fallback_mitigations: Sequence[str] = (),
    fallback_artifacts: Sequence[str] = ("runbook_attached", "rca", "postmortem"),
) -> tuple[List[str], List[str], List[str]]:
    evidence = _preferred(incident.get("required_evidence", []), fallback_evidence)
    mitigations = _preferred(incident.get("required_mitigations", []), fallback_mitigations)
    artifacts = _preferred(list((incident.get("artifact_status") or {}).keys()), fallback_artifacts)
    return evidence, mitigations, artifacts


def allowed_targets_from_mitigations(mitigations: Sequence[str], prefix: str) -> List[str]:
    allowed: List[str] = []
    for item in mitigations:
        if item.startswith(prefix):
            allowed.append(item.replace(prefix, "", 1))
    return allowed


def penalty_total(
    action_history: List[Any],
    incident: Dict[str, Any],
    *,
    allowed_restarts: Sequence[str] = (),
    allowed_rollbacks: Sequence[str] = (),
    ignored_alert_weight: float = 0.03,
) -> float:
    penalty = ignored_alert_weight * incident.get("ignored_alert_ticks", 0)
    allowed_restarts = set(allowed_restarts)
    allowed_rollbacks = set(allowed_rollbacks)
    seen: Dict[tuple[str, Any], int] = {}

    for action in action_history:
        if not isinstance(action, dict):
            continue
        key = (action.get("action"), action.get("target"))
        seen[key] = seen.get(key, 0) + 1

        if allowed_restarts and action.get("action") == "restart_service" and action.get("target") not in allowed_restarts:
            penalty += 0.08
        if allowed_rollbacks and action.get("action") == "rollback_deploy" and action.get("target") not in allowed_rollbacks:
            penalty += 0.08

    for _, count in seen.items():
        if count > 2:
            penalty += 0.02 * (count - 2)

    penalty += 0.04 * len(incident.get("unnecessary_restarts", []))
    return penalty


def grade_incident(
    action_history: List[Any],
    final_state: Dict[str, Any],
    *,
    fallback_services: Sequence[str] = (),
    fallback_zones: Sequence[str] = (),
    fallback_evidence: Sequence[str] = (),
    fallback_mitigations: Sequence[str] = (),
    error_target: float,
    latency_target: int,
    queue_target: int | None = None,
    availability_target: float | None = None,
    service_weight: float = 0.32,
    zone_weight: float = 0.1,
    metric_weight: float = 0.18,
    evidence_weight: float = 0.12,
    mitigation_weight: float = 0.12,
    artifact_weight: float = 0.08,
    communication_weight: float = 0.04,
    sla_weight: float = 0.04,
    ignored_alert_weight: float = 0.03,
) -> float:
    services = final_state.get("services", {})
    metrics = final_state.get("metrics", {})
    incident = final_state.get("incident", {})
    zones = final_state.get("zones", {})
    sla = final_state.get("sla", {})

    target_services, target_zones = dynamic_targets(
        incident,
        fallback_services=fallback_services,
        fallback_zones=fallback_zones,
    )
    required_evidence, required_mitigations, required_artifacts = dynamic_requirements(
        incident,
        fallback_evidence=fallback_evidence,
        fallback_mitigations=fallback_mitigations,
    )

    component_scores: List[tuple[float, float]] = []
    if target_services:
        component_scores.append((service_weight, service_health_ratio(services, target_services)))
    if target_zones:
        component_scores.append((zone_weight, zone_health_ratio(zones, target_zones)))
    component_scores.append(
        (
            metric_weight,
            metric_ratio(
                metrics,
                error_target=error_target,
                latency_target=latency_target,
                queue_target=queue_target,
                availability_target=availability_target,
            ),
        )
    )
    if required_evidence:
        component_scores.append((evidence_weight, evidence_ratio(incident, required_evidence)))
    if required_mitigations:
        component_scores.append((mitigation_weight, mitigation_ratio(incident, required_mitigations)))
    if required_artifacts:
        component_scores.append((artifact_weight, artifact_ratio(incident, required_artifacts)))
    component_scores.append((communication_weight, communication_ratio(incident, action_history)))
    component_scores.append((sla_weight, 1.0 if sla.get("verified") and sla.get("current_status") == "CURRENTLY_MET" else 0.0))

    total_weight = sum(weight for weight, _ in component_scores)
    raw_score = 0.0 if total_weight <= 0 else sum(weight * score for weight, score in component_scores) / total_weight

    allowed_restarts = allowed_targets_from_mitigations(required_mitigations, "restart_service:")
    allowed_rollbacks = allowed_targets_from_mitigations(required_mitigations, "rollback_deploy:")
    penalty = penalty_total(
        action_history,
        incident,
        allowed_restarts=allowed_restarts,
        allowed_rollbacks=allowed_rollbacks,
        ignored_alert_weight=ignored_alert_weight,
    )
    return clamp_score(raw_score - penalty)
