from typing import Dict, Any
from .models import ServiceState, Scenario, Alert

def _build_base_logs():
    return [f"[INFO] System ops normal at tick {-i}" for i in range(50, 0, -1)]

SCENARIOS: Dict[Scenario, Dict[str, Any]] = {
    Scenario.Easy: {
        'services': {
            'database': ServiceState.down,
            'api': ServiceState.healthy,
            'frontend': ServiceState.healthy,
            'worker': ServiceState.healthy,
            'auth': ServiceState.healthy
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
                message="Connection refused on port 5432",
                fired_at=0
            )
        ],
        'logs': _build_base_logs() + ["[ERROR] Connection refused on port 5432"],
        'deploy_history': ["Deploy v1.0.0 (success)", "Deploy v1.0.1 (success)"],
        'code_diffs': ["- init_db_connection()", "+ init_db_connection(timeout=2)"],
        'sla_status': "CURRENTLY_MET",
        'estimated_affected_users': 0
    },
    Scenario.Medium: {
        'services': {
            'database': ServiceState.healthy,
            'api': ServiceState.healthy,
            'frontend': ServiceState.healthy,
            'worker': ServiceState.degraded,
            'auth': ServiceState.healthy
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
            'database': ServiceState.degraded,
            'api': ServiceState.healthy,
            'frontend': ServiceState.healthy,
            'worker': ServiceState.healthy,
            'auth': ServiceState.healthy
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
            "2026-04-01 19:00:00 — api-service v2.3.0 deployed (success)",
            "2026-04-01 20:31:00 — api-service v2.3.1 deployed (success)",
            "2026-04-01 19:45:00 — worker-service v1.8.2 deployed (success)",
        ],
        'code_diffs': [
            "--- api-service/db/queries.py (v2.3.0 -> v2.3.1)",
            "- SELECT * FROM orders WHERE user_id = $1 AND created_at > $2 LIMIT 100",
            "+ SELECT * FROM orders WHERE user_id = $1  -- removed date filter and LIMIT",
            "  # NOTE: v2.3.1 removed the index-friendly predicate, causing full table scan",
        ],
        'sla_status': "CURRENTLY_MET",
        'estimated_affected_users': 0
    }
}
