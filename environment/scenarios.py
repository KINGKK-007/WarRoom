import copy
import random
from typing import Any, Dict, List, Optional

from .models import Alert, AvailabilityZone, Scenario, ServiceState


ZONES = [zone.value for zone in AvailabilityZone]


SERVICE_DEPENDENCIES: Dict[str, List[str]] = {
    "postgres-primary": [],
    "postgres-replica": ["postgres-primary"],
    "redis-cache": [],
    "redis-session": [],
    "kafka": [],
    "zookeeper": [],
    "mongodb": [],
    "clickhouse": [],
    "elasticsearch": [],
    "object-storage": [],
    "config-service": [],
    "service-mesh": [],
    "api-gateway": ["edge-proxy", "service-mesh", "auth-service"],
    "edge-proxy": ["service-mesh"],
    "auth-service": ["redis-session", "postgres-primary"],
    "user-service": ["postgres-primary", "redis-cache"],
    "profile-service": ["user-service", "mongodb"],
    "billing-service": ["postgres-primary", "kafka"],
    "order-service": ["postgres-primary", "inventory-service", "payment-service"],
    "inventory-service": ["postgres-primary", "redis-cache"],
    "payment-service": ["billing-service", "fraud-service"],
    "cart-service": ["redis-cache", "user-service"],
    "recommendation-service": ["clickhouse", "kafka"],
    "search-service": ["elasticsearch"],
    "notification-service": ["kafka", "email-service"],
    "email-service": ["object-storage"],
    "analytics-service": ["kafka", "clickhouse"],
    "report-service": ["analytics-service", "clickhouse"],
    "worker-service": ["kafka", "redis-cache"],
    "scheduler-service": ["worker-service", "config-service"],
    "fraud-service": ["postgres-primary", "mongodb"],
    "frontend-web": ["api-gateway", "search-service"],
    "mobile-bff": ["api-gateway", "user-service"],
    "admin-portal": ["api-gateway", "report-service"],
    "prometheus": [],
    "grafana": ["prometheus"],
    "loki": ["object-storage"],
    "tempo": ["object-storage"],
    "status-page": [],
}


SERVICE_TIERS: Dict[str, str] = {
    "postgres-primary": "data",
    "postgres-replica": "data",
    "redis-cache": "data",
    "redis-session": "data",
    "kafka": "data",
    "zookeeper": "infra",
    "mongodb": "data",
    "clickhouse": "data",
    "elasticsearch": "data",
    "object-storage": "infra",
    "config-service": "infra",
    "service-mesh": "infra",
    "api-gateway": "edge",
    "edge-proxy": "edge",
    "auth-service": "app",
    "user-service": "app",
    "profile-service": "app",
    "billing-service": "app",
    "order-service": "app",
    "inventory-service": "app",
    "payment-service": "app",
    "cart-service": "app",
    "recommendation-service": "app",
    "search-service": "app",
    "notification-service": "app",
    "email-service": "app",
    "analytics-service": "app",
    "report-service": "app",
    "worker-service": "app",
    "scheduler-service": "app",
    "fraud-service": "app",
    "frontend-web": "edge",
    "mobile-bff": "edge",
    "admin-portal": "edge",
    "prometheus": "observability",
    "grafana": "observability",
    "loki": "observability",
    "tempo": "observability",
    "status-page": "ops",
}


CHAOS_SERVICE_FAILURES = [
    ("postgres-primary", "restart_service"),
    ("worker-service", "clear_queue"),
    ("api-gateway", "rollback_deploy"),
    ("redis-cache", "restart_service"),
    ("service-mesh", "restart_service"),
    ("search-service", "restart_service"),
    ("kafka", "restart_service"),
    ("order-service", "restart_service"),
]

CHAOS_ZONE_FAILURES = ["us-east-1a", "us-east-1b", "us-east-1c", "us-central-1a"]
CHAOS_DEPLOY_FAILURES = [("api-gateway", "v3.2.0"), ("worker-service", "v3.2.0"), ("frontend-web", "v5.0.4")]


def _service_detail(name: str) -> Dict[str, Any]:
    default_version = "v3.2.0" if SERVICE_TIERS[name] == "app" else "v1.0.0"
    if name in {"frontend-web", "mobile-bff", "admin-portal"}:
        default_version = "v5.0.4"
    return {
        "tier": SERVICE_TIERS[name],
        "dependencies": copy.deepcopy(SERVICE_DEPENDENCIES.get(name, [])),
        "version": default_version,
        "replicas": 3 if SERVICE_TIERS[name] in {"app", "edge"} else 2,
        "zones": list(ZONES if name not in {"postgres-primary", "zookeeper"} else ZONES[:3]),
        "zone_states": {zone: ServiceState.healthy for zone in (ZONES if name not in {"postgres-primary", "zookeeper"} else ZONES[:3])},
        "queue_depth": 0,
        "cpu": 0.42,
        "memory": 0.45,
        "traffic_weight": 1.0,
        "autoscaling_tuned": False,
        "isolated": False,
        "health_checks": 0,
    }


def _build_logs() -> List[Dict[str, Any]]:
    return [
        {
            "tick": -25 + idx,
            "service": "platform",
            "level": "INFO",
            "message": f"Steady-state health check {idx} passed",
        }
        for idx in range(1, 25)
    ]


def _build_traces() -> List[Dict[str, Any]]:
    return [
        {"trace_id": "trc-baseline-01", "service": "frontend-web", "span": "GET /checkout", "latency_ms": 180, "status": "ok", "zone": "us-east-1a"},
        {"trace_id": "trc-baseline-02", "service": "api-gateway", "span": "POST /orders", "latency_ms": 210, "status": "ok", "zone": "us-east-1b"},
    ]


def _base_incident() -> Dict[str, Any]:
    return {
        "name": "baseline",
        "summary": "No active incident",
        "root_causes": [],
        "service_targets": [],
        "zone_targets": [],
        "deploy_targets": {},
        "required_evidence": [],
        "required_mitigations": [],
        "required_production_actions": ["verify_sla", "attach_runbook", "run_rca", "generate_postmortem"],
        "evidence_collected": [],
        "mitigations_applied": [],
        "production_actions_completed": [],
        "actions_executed": [],
        "artifact_status": {"rca": False, "postmortem": False, "sla_verified": False, "runbook_attached": False},
        "runbook_attached": False,
        "status_page_updated": False,
        "escalated_to": [],
        "notified": [],
        "acknowledged_alerts": False,
        "rca": None,
        "postmortem": None,
        "resolved": True,
        "severity": "none",
        "recovery_tick": None,
        "timeline": [],
        "events": [],
        "ignored_alert_ticks": 0,
        "unnecessary_restarts": [],
        "pending_failures": [],
        "network_partitions": [],
        "needs_manager_comms": False,
    }


def _build_base_state() -> Dict[str, Any]:
    details = {name: _service_detail(name) for name in SERVICE_DEPENDENCIES}
    return {
        "services": {name: ServiceState.healthy for name in SERVICE_DEPENDENCIES},
        "service_details": details,
        "service_distribution": {name: copy.deepcopy(detail["zones"]) for name, detail in details.items()},
        "zones": {zone: {"status": "healthy", "packet_loss": 0.0, "latency_ms": 12, "drained": False, "failed_over": False} for zone in ZONES},
        "metrics": {
            "error_rate": 0.012,
            "cpu": 0.41,
            "memory": 0.49,
            "p99_latency_ms": 185,
            "requests_per_sec": 1450,
            "queue_depth": 12,
            "availability": 99.97,
            "saturation": 0.38,
        },
        "metrics_history": [],
        "alerts": [],
        "logs": _build_logs(),
        "historical_logs": [],
        "traces": _build_traces(),
        "deploy_history": [
            {"service": "api-gateway", "version": "v3.2.0", "tick": -8, "status": "success"},
            {"service": "worker-service", "version": "v3.2.0", "tick": -6, "status": "success"},
            {"service": "frontend-web", "version": "v5.0.4", "tick": -3, "status": "success"},
        ],
        "code_diffs": ["api-gateway routing changes stable", "worker-service concurrency steady"],
        "sla": {
            "target_availability": 99.9,
            "target_error_rate": 0.02,
            "current_status": "CURRENTLY_MET",
            "breaches": [],
            "verified": False,
        },
        "estimated_affected_users": 0,
        "incident": _base_incident(),
    }


def _append_incident_timeline(state: Dict[str, Any], entry: str):
    state["incident"]["timeline"].append(entry)


def _append_incident_event(state: Dict[str, Any], event_type: str, summary: str, payload: Optional[Dict[str, Any]] = None):
    state["incident"].setdefault("events", []).append(
        {
            "tick": 0,
            "type": event_type,
            "summary": summary,
            "payload": payload or {},
        }
    )


def _easy_scenario() -> Dict[str, Any]:
    state = _build_base_state()
    for zone in state["service_details"]["postgres-primary"]["zones"]:
        state["service_details"]["postgres-primary"]["zone_states"][zone] = ServiceState.down
    state["services"]["postgres-primary"] = ServiceState.down
    state["services"]["postgres-replica"] = ServiceState.degraded
    state["services"]["user-service"] = ServiceState.degraded
    state["services"]["billing-service"] = ServiceState.degraded
    state["services"]["auth-service"] = ServiceState.degraded
    state["metrics"].update({"error_rate": 0.18, "p99_latency_ms": 860, "requests_per_sec": 540, "availability": 97.8, "saturation": 0.81})
    state["alerts"] = [
        Alert(severity="CRITICAL", message="postgres-primary unreachable across all replicas", fired_at=0, source="postgres-primary"),
        Alert(severity="CRITICAL", message="Auth and billing elevated dependency failures", fired_at=0, source="auth-service"),
    ]
    state["logs"].extend(
        [
            {"tick": 0, "service": "postgres-primary", "level": "ERROR", "message": "connection refused on 5432"},
            {"tick": 0, "service": "auth-service", "level": "ERROR", "message": "session validation blocked by primary database outage"},
            {"tick": 0, "service": "billing-service", "level": "ERROR", "message": "cannot persist payment intent"},
        ]
    )
    state["traces"].append({"trace_id": "trc-easy-01", "service": "auth-service", "span": "POST /login", "latency_ms": 1220, "status": "db_error", "zone": "us-east-1a"})
    state["estimated_affected_users"] = 4200
    state["incident"].update(
        {
            "name": "primary-database-outage",
            "summary": "Primary database outage is cascading into authentication and billing.",
            "root_causes": ["postgres-primary"],
            "service_targets": ["postgres-primary"],
            "zone_targets": [],
            "deploy_targets": {},
            "required_evidence": ["inspect:postgres-primary", "query_metrics", "query_logs", "run_health_check:postgres-primary"],
            "required_mitigations": ["restart_service:postgres-primary"],
            "resolved": False,
            "severity": "sev1",
            "needs_manager_comms": True,
        }
    )
    _append_incident_timeline(state, "tick=0 incident opened: primary database outage")
    _append_incident_event(state, "incident_opened", "Primary database outage detected.", {"root_causes": ["postgres-primary"]})
    return state


def _medium_scenario() -> Dict[str, Any]:
    state = _build_base_state()
    state["service_details"]["worker-service"]["queue_depth"] = 24000
    state["service_details"]["worker-service"]["cpu"] = 0.94
    state["service_details"]["worker-service"]["memory"] = 0.91
    state["services"]["worker-service"] = ServiceState.degraded
    state["services"]["scheduler-service"] = ServiceState.degraded
    state["services"]["notification-service"] = ServiceState.degraded
    state["services"]["analytics-service"] = ServiceState.degraded
    state["zones"]["us-east-1b"]["status"] = "degraded"
    state["zones"]["us-east-1b"]["packet_loss"] = 0.08
    state["zones"]["us-east-1b"]["latency_ms"] = 65
    state["metrics"].update({"error_rate": 0.09, "p99_latency_ms": 640, "requests_per_sec": 980, "queue_depth": 24000, "availability": 99.1, "saturation": 0.88, "cpu": 0.83})
    state["alerts"] = [
        Alert(severity="WARNING", message="worker-service backlog above 20k", fired_at=0, source="worker-service"),
        Alert(severity="CRITICAL", message="scheduler-service lagging due to worker backlog", fired_at=0, source="scheduler-service"),
    ]
    state["logs"].extend(
        [
            {"tick": 0, "service": "worker-service", "level": "WARN", "message": "queue depth climbing to 24000"},
            {"tick": 0, "service": "worker-service", "level": "ERROR", "message": "worker concurrency exhausted"},
            {"tick": 0, "service": "scheduler-service", "level": "WARN", "message": "dispatch lag crossed 6 minutes"},
        ]
    )
    state["traces"].append({"trace_id": "trc-medium-01", "service": "notification-service", "span": "fanout notifications", "latency_ms": 980, "status": "queue_backlog", "zone": "us-east-1b"})
    state["estimated_affected_users"] = 2800
    state["incident"].update(
        {
            "name": "queue-backlog-and-zone-degradation",
            "summary": "Worker backlog and partial zone impairment are degrading asynchronous services.",
            "root_causes": ["worker-service"],
            "service_targets": ["worker-service"],
            "zone_targets": ["us-east-1b"],
            "deploy_targets": {},
            "required_evidence": ["inspect:worker-service", "query_metrics", "query_logs", "query_traces", "query_topology"],
            "required_mitigations": ["clear_queue:worker-service", "scale:worker-service", "tune_autoscaling:worker-service", "drain_zone:us-east-1b", "restore_zone:us-east-1b"],
            "resolved": False,
            "severity": "sev2",
            "pending_failures": [{"kind": "worker_oom", "service": "worker-service", "trigger_tick": 2}],
        }
    )
    _append_incident_timeline(state, "tick=0 incident opened: worker backlog and zone degradation")
    _append_incident_event(state, "incident_opened", "Worker backlog and zone degradation detected.", {"root_causes": ["worker-service", "us-east-1b"]})
    return state


def _hard_scenario() -> Dict[str, Any]:
    state = _build_base_state()
    state["service_details"]["api-gateway"]["version"] = "v3.2.1"
    state["code_diffs"] = [
        "--- api-gateway/routing.py",
        "- timeout_budget_ms = 300",
        "+ timeout_budget_ms = 1800",
        "+ retry_fanout = 4  # amplifies requests under packet loss",
    ]
    state["deploy_history"].append({"service": "api-gateway", "version": "v3.2.1", "tick": -1, "status": "success"})
    state["zones"]["us-east-1c"]["status"] = "down"
    state["zones"]["us-east-1c"]["packet_loss"] = 0.38
    state["zones"]["us-east-1c"]["latency_ms"] = 320
    state["service_details"]["api-gateway"]["zone_states"]["us-east-1c"] = ServiceState.down
    state["service_details"]["frontend-web"]["zone_states"]["us-east-1c"] = ServiceState.degraded
    state["services"]["api-gateway"] = ServiceState.degraded
    state["services"]["frontend-web"] = ServiceState.degraded
    state["services"]["mobile-bff"] = ServiceState.degraded
    state["services"]["order-service"] = ServiceState.degraded
    state["metrics"].update({"error_rate": 0.16, "p99_latency_ms": 2140, "requests_per_sec": 720, "availability": 98.4, "saturation": 0.92, "cpu": 0.79})
    state["alerts"] = [
        Alert(severity="CRITICAL", message="Gateway latency and retries spiking after api-gateway v3.2.1 deploy", fired_at=0, source="api-gateway"),
        Alert(severity="CRITICAL", message="Zone us-east-1c packet loss > 35%", fired_at=0, source="us-east-1c"),
    ]
    state["logs"].extend(
        [
            {"tick": 0, "service": "api-gateway", "level": "WARN", "message": "retry storm detected after deployment v3.2.1"},
            {"tick": 0, "service": "service-mesh", "level": "ERROR", "message": "east-1c upstream packet loss causing gateway retries"},
            {"tick": 0, "service": "frontend-web", "level": "WARN", "message": "checkout failures concentrated in east-1c"},
        ]
    )
    state["traces"].extend(
        [
            {"trace_id": "trc-hard-01", "service": "api-gateway", "span": "POST /orders", "latency_ms": 2450, "status": "retry_storm", "zone": "us-east-1c"},
            {"trace_id": "trc-hard-02", "service": "frontend-web", "span": "GET /checkout", "latency_ms": 1960, "status": "upstream_timeout", "zone": "us-east-1c"},
        ]
    )
    state["estimated_affected_users"] = 11800
    state["incident"].update(
        {
            "name": "bad-gateway-deploy-with-zone-failure",
            "summary": "A bad api-gateway deploy is amplifying a zone failure into a customer-facing outage.",
            "root_causes": ["api-gateway", "us-east-1c"],
            "service_targets": ["api-gateway"],
            "zone_targets": ["us-east-1c"],
            "deploy_targets": {"api-gateway": "v3.2.0"},
            "required_evidence": ["query_metrics", "query_traces", "query_deploy", "query_topology", "inspect:api-gateway"],
            "required_mitigations": ["rollback_deploy:api-gateway", "failover_zone:us-east-1c", "rebalance_traffic:api-gateway"],
            "resolved": False,
            "severity": "sev1",
            "needs_manager_comms": True,
        }
    )
    _append_incident_timeline(state, "tick=0 incident opened: bad deploy plus zone outage")
    _append_incident_event(state, "incident_opened", "Bad api-gateway deploy and zone outage detected.", {"root_causes": ["api-gateway", "us-east-1c"]})
    return state


def generate_procedural_incident(seed: int = 20260407) -> Dict[str, Any]:
    rng = random.Random(seed)
    state = _build_base_state()
    chosen_services = rng.sample(CHAOS_SERVICE_FAILURES, 3)
    chosen_zones = rng.sample(CHAOS_ZONE_FAILURES, 2)
    chosen_deploy = rng.choice(CHAOS_DEPLOY_FAILURES)

    root_causes: List[str] = []
    required_evidence = {"query_metrics", "query_logs", "query_traces", "query_topology"}
    required_mitigations = set()
    deploy_targets: Dict[str, str] = {}

    for service, preferred_action in chosen_services:
        root_causes.append(service)
        state["services"][service] = ServiceState.down if preferred_action == "restart_service" else ServiceState.degraded
        for zone in state["service_details"][service]["zone_states"]:
            state["service_details"][service]["zone_states"][zone] = state["services"][service]
        state["alerts"].append(Alert(severity="CRITICAL", message=f"Chaos injected failure on {service}", fired_at=0, source=service))
        state["logs"].append({"tick": 0, "service": service, "level": "ERROR", "message": f"chaos: {service} entered {state['services'][service].value}"})
        state["traces"].append({"trace_id": f"chaos-{service}", "service": service, "span": "chaos-inject", "latency_ms": 2600, "status": "chaos_failure", "zone": state["service_details"][service]["zones"][0]})
        required_evidence.add(f"inspect:{service}")
        if preferred_action == "restart_service":
            required_mitigations.add(f"restart_service:{service}")
        elif preferred_action == "clear_queue":
            state["service_details"][service]["queue_depth"] = 28000
            required_mitigations.update({f"clear_queue:{service}", f"scale:{service}", f"tune_autoscaling:{service}"})
        elif preferred_action == "rollback_deploy":
            broken_version = "v9.9.9-chaos"
            state["service_details"][service]["version"] = broken_version
            state["deploy_history"].append({"service": service, "version": broken_version, "tick": -1, "status": "success"})
            deploy_targets[service] = chosen_deploy[1] if service == chosen_deploy[0] else "v3.2.0"
            required_mitigations.add(f"rollback_deploy:{service}")

    for zone in chosen_zones:
        root_causes.append(zone)
        state["zones"][zone]["status"] = "down"
        state["zones"][zone]["packet_loss"] = round(rng.uniform(0.22, 0.51), 2)
        state["zones"][zone]["latency_ms"] = rng.randint(180, 420)
        state["incident"]["network_partitions"].append(zone)
        state["alerts"].append(Alert(severity="CRITICAL", message=f"Chaos network partition in {zone}", fired_at=0, source=zone))
        required_mitigations.update({f"failover_zone:{zone}", f"restore_zone:{zone}"})

    deploy_service, good_version = chosen_deploy
    root_causes.append(f"deploy:{deploy_service}")
    state["service_details"][deploy_service]["version"] = "v9.9.9-chaos"
    state["services"][deploy_service] = ServiceState.degraded
    for zone in state["service_details"][deploy_service]["zone_states"]:
        if state["zones"][zone]["status"] == "healthy":
            state["service_details"][deploy_service]["zone_states"][zone] = ServiceState.degraded
    state["deploy_history"].append({"service": deploy_service, "version": "v9.9.9-chaos", "tick": -1, "status": "success"})
    deploy_targets[deploy_service] = good_version
    required_evidence.add(f"inspect:{deploy_service}")
    required_evidence.add("query_deploy")
    required_mitigations.update({f"rollback_deploy:{deploy_service}", f"rebalance_traffic:{deploy_service}"})

    state["incident"].update(
        {
            "name": "chaos-multi-root-cause",
            "summary": "Procedurally generated overlapping storage, deploy, and network failures.",
            "root_causes": root_causes,
            "service_targets": sorted({svc for svc, _ in chosen_services} | {deploy_service}),
            "zone_targets": chosen_zones,
            "deploy_targets": deploy_targets,
            "required_evidence": sorted(required_evidence),
            "required_mitigations": sorted(required_mitigations),
            "resolved": False,
            "severity": "sev0",
            "needs_manager_comms": True,
        }
    )
    state["metrics"].update({"error_rate": 0.31, "p99_latency_ms": 2860, "requests_per_sec": 380, "availability": 95.2, "saturation": 0.97, "queue_depth": 36000})
    state["estimated_affected_users"] = 22400
    _append_incident_timeline(state, f"tick=0 chaos incident generated seed={seed}")
    _append_incident_event(state, "incident_opened", "Procedural chaos incident generated.", {"seed": seed, "root_causes": root_causes})
    return state


def generate_variant(task_id: Scenario, seed: int) -> Dict[str, Any]:
    if task_id == Scenario.CHAOS:
        return generate_procedural_incident(seed=seed)

    rng = random.Random(seed)
    if task_id == Scenario.EASY:
        state = _easy_scenario()
        impacted = rng.choice(["auth-service", "billing-service", "user-service"])
        state["services"][impacted] = ServiceState.degraded
        state["logs"].append({"tick": 0, "service": impacted, "level": "ERROR", "message": f"variant cascade observed in {impacted}"})
        state["incident"]["summary"] = f"Seeded variant: postgres-primary outage with elevated impact on {impacted}."
        state["incident"]["variant"] = {"seed": seed, "focus_service": impacted}
        return state

    if task_id == Scenario.MEDIUM:
        state = _medium_scenario()
        zone = rng.choice(["us-east-1a", "us-east-1b", "us-east-1c"])
        queue_depth = rng.choice([18000, 22000, 30000])
        state["zones"]["us-east-1b"]["status"] = "healthy"
        state["zones"]["us-east-1b"]["packet_loss"] = 0.0
        state["zones"]["us-east-1b"]["latency_ms"] = 12
        state["zones"][zone]["status"] = "degraded"
        state["zones"][zone]["packet_loss"] = 0.06
        state["zones"][zone]["latency_ms"] = 58
        state["service_details"]["worker-service"]["queue_depth"] = queue_depth
        state["incident"]["zone_targets"] = [zone]
        state["incident"]["required_mitigations"] = [
            "clear_queue:worker-service",
            "scale:worker-service",
            "tune_autoscaling:worker-service",
            f"drain_zone:{zone}",
            f"restore_zone:{zone}",
        ]
        state["incident"]["summary"] = f"Seeded variant: worker backlog with zone degradation in {zone}."
        state["incident"]["variant"] = {"seed": seed, "focus_zone": zone, "queue_depth": queue_depth}
        return state

    if task_id == Scenario.HARD:
        state = _hard_scenario()
        zone = rng.choice(["us-east-1a", "us-east-1b", "us-east-1c"])
        deploy_target = rng.choice(["api-gateway", "frontend-web"])
        good_version = "v3.2.0" if deploy_target == "api-gateway" else "v5.0.4"
        for z in state["zones"]:
            state["zones"][z]["status"] = "healthy"
            state["zones"][z]["packet_loss"] = 0.0
            state["zones"][z]["latency_ms"] = 12
        state["zones"][zone]["status"] = "down"
        state["zones"][zone]["packet_loss"] = 0.34
        state["zones"][zone]["latency_ms"] = 280
        state["incident"]["zone_targets"] = [zone]
        state["incident"]["deploy_targets"] = {deploy_target: good_version}
        if deploy_target != "api-gateway":
            state["service_details"]["frontend-web"]["version"] = "v5.0.5-bad"
            state["services"]["frontend-web"] = ServiceState.degraded
            state["incident"]["service_targets"] = [deploy_target]
            state["incident"]["required_evidence"] = ["query_metrics", "query_traces", "query_deploy", "query_topology", "inspect:frontend-web"]
            state["incident"]["required_mitigations"] = [f"rollback_deploy:{deploy_target}", f"failover_zone:{zone}", f"rebalance_traffic:{deploy_target}"]
        else:
            state["service_details"]["api-gateway"]["zone_states"][zone] = ServiceState.down
            state["incident"]["required_mitigations"] = [f"rollback_deploy:{deploy_target}", f"failover_zone:{zone}", f"rebalance_traffic:{deploy_target}"]
        state["incident"]["summary"] = f"Seeded variant: bad deploy on {deploy_target} with zone outage in {zone}."
        state["incident"]["variant"] = {"seed": seed, "focus_zone": zone, "deploy_target": deploy_target}
        return state

    return copy.deepcopy(SCENARIOS[task_id])


SCENARIOS: Dict[Scenario, Dict[str, Any]] = {
    Scenario.EASY: _easy_scenario(),
    Scenario.MEDIUM: _medium_scenario(),
    Scenario.HARD: _hard_scenario(),
    Scenario.CHAOS: generate_procedural_incident(),
}
