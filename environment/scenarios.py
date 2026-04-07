from typing import Dict, Any, List, Tuple
from .models import ServiceState, Scenario, Alert

def _build_base_logs():
    return [f"[INFO] System ops normal at tick {-i}" for i in range(50, 0, -1)]

# Service dependency graph: (service_name, [dependencies])
# If any dependency is DOWN, this service will degrade
SERVICE_DEPENDENCIES: Dict[str, List[str]] = {
    # Data Tier (no dependencies - foundational services)
    'postgres': [],
    'redis': [],
    'elasticsearch': [],
    'kafka': [],
    'mongodb': [],

    # Application Tier
    'api-gateway': ['nginx'],
    'api-core': ['postgres', 'redis'],
    'user-service': ['postgres'],
    'payment-service': ['postgres', 'kafka'],
    'notification-service': ['kafka'],
    'worker': ['redis', 'kafka'],

    # Frontend/Edge
    'nginx': [],
    'frontend': ['api-gateway'],

    # Infrastructure
    'auth': ['redis'],
    'prometheus': [],
    'grafana': ['prometheus'],
}

SCENARIOS: Dict[Scenario, Dict[str, Any]] = {
    Scenario.Easy: {
        'services': {
            # Data Tier
            'postgres': ServiceState.down,
            'redis': ServiceState.healthy,
            'elasticsearch': ServiceState.healthy,
            'kafka': ServiceState.healthy,
            'mongodb': ServiceState.healthy,

            # Application Tier
            'api-gateway': ServiceState.healthy,
            'api-core': ServiceState.healthy,
            'user-service': ServiceState.healthy,
            'payment-service': ServiceState.healthy,
            'notification-service': ServiceState.healthy,
            'worker': ServiceState.healthy,

            # Frontend/Edge
            'nginx': ServiceState.healthy,
            'frontend': ServiceState.healthy,

            # Infrastructure
            'auth': ServiceState.healthy,
            'prometheus': ServiceState.healthy,
            'grafana': ServiceState.healthy,
        },
        'metrics': {
            'error_rate': 0.0,
            'cpu': 0.45,
            'memory': 0.52,
            'p99_latency_ms': 120,
            'requests_per_sec': 850,
        },
        'alerts': [
            Alert(
                severity="CRITICAL",
                message="PostgreSQL connection refused on port 5432",
                fired_at=0
            )
        ],
        'logs': _build_base_logs() + [
            "[ERROR] PostgreSQL connection refused on port 5432",
            "[ERROR] api-core: Database connection pool exhausted",
            "[ERROR] user-service: Cannot connect to postgres"
        ],
        'deploy_history': ["Deploy v1.0.0 (success)", "Deploy v1.0.1 (success)"],
        'code_diffs': ["- init_db_connection()", "+ init_db_connection(timeout=2)"],
        'sla_status': "CURRENTLY_MET",
        'estimated_affected_users': 0
    },
    Scenario.Medium: {
        'services': {
            # Data Tier
            'postgres': ServiceState.healthy,
            'redis': ServiceState.healthy,
            'elasticsearch': ServiceState.healthy,
            'kafka': ServiceState.healthy,
            'mongodb': ServiceState.healthy,

            # Application Tier
            'api-gateway': ServiceState.healthy,
            'api-core': ServiceState.healthy,
            'user-service': ServiceState.healthy,
            'payment-service': ServiceState.healthy,
            'notification-service': ServiceState.healthy,
            'worker': ServiceState.degraded,

            # Frontend/Edge
            'nginx': ServiceState.healthy,
            'frontend': ServiceState.healthy,

            # Infrastructure
            'auth': ServiceState.healthy,
            'prometheus': ServiceState.healthy,
            'grafana': ServiceState.healthy,
        },
        'metrics': {
            'error_rate': 0.05,
            'cpu': 0.78,
            'memory': 0.78,
            'p99_latency_ms': 145,
            'requests_per_sec': 620,
        },
        'alerts': [],
        'logs': _build_base_logs() + [
            "[WARN] Memory usage at 78%",
            "[WARN] worker-service: Memory usage climbing — 78% (threshold: 80%)",
            "[ERROR] worker-service: OOM kill imminent — process unstable",
        ],
        'deploy_history': ["Deploy v1.0.0 (success)", "Deploy v1.0.1 (success)"],
        'code_diffs': ["- init_db_connection()", "+ init_db_connection(timeout=2)"],
        'sla_status': "CURRENTLY_MET",
        'estimated_affected_users': 0
    },
    Scenario.Hard: {
        'services': {
            # Data Tier
            'postgres': ServiceState.degraded,
            'redis': ServiceState.healthy,
            'elasticsearch': ServiceState.healthy,
            'kafka': ServiceState.healthy,
            'mongodb': ServiceState.healthy,

            # Application Tier
            'api-gateway': ServiceState.healthy,
            'api-core': ServiceState.healthy,
            'user-service': ServiceState.healthy,
            'payment-service': ServiceState.healthy,
            'notification-service': ServiceState.healthy,
            'worker': ServiceState.healthy,

            # Frontend/Edge
            'nginx': ServiceState.healthy,
            'frontend': ServiceState.healthy,

            # Infrastructure
            'auth': ServiceState.healthy,
            'prometheus': ServiceState.healthy,
            'grafana': ServiceState.healthy,
        },
        'metrics': {
            'error_rate': 0.10,
            'cpu': 0.62,
            'memory': 0.55,
            'p99_latency_ms': 1840,
            'requests_per_sec': 310,
        },
        'alerts': [],
        'logs': _build_base_logs() + [
            "[WARN] p99 latency trending up: 120ms -> 134ms -> 167ms -> 1840ms",
            "[WARN] db_query_time trending up: 12ms -> 26ms -> 38ms -> 55ms",
            "[INFO] No critical alerts firing — all SLAs currently met",
        ],
        'deploy_history': [
            "2026-04-01 19:00:00 — api-core v2.3.0 deployed (success)",
            "2026-04-01 20:31:00 — api-core v2.3.1 deployed (success)",
            "2026-04-01 19:45:00 — worker v1.8.2 deployed (success)",
            "2026-04-01 18:30:00 — payment-service v3.1.0 deployed (success)",
            "2026-04-01 17:15:00 — user-service v2.0.1 deployed (success)",
        ],
        'code_diffs': [
            "--- api-core/db/queries.py (v2.3.0 -> v2.3.1)",
            "- SELECT * FROM orders WHERE user_id = $1 AND created_at > $2 LIMIT 100",
            "+ SELECT * FROM orders WHERE user_id = $1  -- removed date filter and LIMIT",
            "  # NOTE: v2.3.1 removed the index-friendly predicate, causing full table scan",
        ],
        'sla_status': "CURRENTLY_MET",
        'estimated_affected_users': 0
    }
}
