from graders import chaos, task_1, task_2, task_3, task_4, task_5, task_6, task_7, task_8, task_9, task_10, task_11


def test_task_1_golden_path_scores_precisely_one():
    final_state = {
        "services": {"postgres-primary": "healthy", "auth-service": "healthy", "billing-service": "healthy"},
        "metrics": {"error_rate": 0.03, "p99_latency_ms": 320},
        "sla": {"verified": True, "current_status": "CURRENTLY_MET"},
        "incident": {
            "evidence_collected": ["inspect:postgres-primary", "query_metrics", "query_logs", "run_health_check:postgres-primary"],
            "mitigations_applied": ["restart_service:postgres-primary"],
            "artifact_status": {"rca": True, "postmortem": True, "runbook_attached": True},
            "unnecessary_restarts": [],
            "ignored_alert_ticks": 0,
        },
    }
    history = [{"action": "restart_service", "target": "postgres-primary"}]
    assert task_1.grade(history, final_state) == 1.0


def test_task_1_penalizes_wrong_restart():
    final_state = {
        "services": {"postgres-primary": "healthy", "auth-service": "healthy", "billing-service": "healthy"},
        "metrics": {"error_rate": 0.03, "p99_latency_ms": 320},
        "sla": {"verified": True, "current_status": "CURRENTLY_MET"},
        "incident": {
            "evidence_collected": ["inspect:postgres-primary", "query_metrics", "query_logs", "run_health_check:postgres-primary"],
            "mitigations_applied": ["restart_service:postgres-primary"],
            "artifact_status": {"rca": True, "postmortem": True, "runbook_attached": True},
            "unnecessary_restarts": ["api-gateway"],
            "ignored_alert_ticks": 0,
        },
    }
    history = [{"action": "restart_service", "target": "api-gateway"}]
    assert task_1.grade(history, final_state) < 0.9


def test_task_2_state_based_grader_requires_real_recovery():
    final_state = {
        "services": {"worker-service": "healthy", "scheduler-service": "healthy", "notification-service": "healthy"},
        "zones": {"us-east-1b": {"status": "healthy"}},
        "metrics": {"queue_depth": 0, "availability": 99.92, "error_rate": 0.03, "p99_latency_ms": 320},
        "incident": {
            "evidence_collected": ["inspect:worker-service", "query_metrics", "query_logs", "query_traces", "query_topology"],
            "mitigations_applied": ["clear_queue:worker-service", "scale:worker-service", "tune_autoscaling:worker-service", "drain_zone:us-east-1b", "restore_zone:us-east-1b"],
            "artifact_status": {"rca": True, "postmortem": True, "runbook_attached": True},
            "ignored_alert_ticks": 0,
        },
    }
    assert task_2.grade([], final_state) >= 0.9


def test_task_3_state_based_grader_requires_zone_and_deploy_recovery():
    final_state = {
        "services": {"api-gateway": "healthy", "frontend-web": "healthy", "mobile-bff": "healthy"},
        "zones": {"us-east-1c": {"status": "healthy"}},
        "metrics": {"p99_latency_ms": 420, "error_rate": 0.02},
        "sla": {"verified": True, "current_status": "CURRENTLY_MET"},
        "incident": {
            "evidence_collected": ["query_metrics", "query_traces", "query_deploy", "query_topology", "inspect:api-gateway"],
            "mitigations_applied": ["rollback_deploy:api-gateway", "failover_zone:us-east-1c", "rebalance_traffic:api-gateway"],
            "artifact_status": {"rca": True, "postmortem": True, "runbook_attached": True},
            "ignored_alert_ticks": 0,
        },
    }
    assert task_3.grade([], final_state) >= 0.9


def test_task_4_state_based_grader_requires_session_recovery():
    final_state = {
        "services": {"redis-session": "healthy", "auth-service": "healthy", "cart-service": "healthy"},
        "metrics": {"error_rate": 0.03, "p99_latency_ms": 340},
        "sla": {"verified": True, "current_status": "CURRENTLY_MET"},
        "incident": {
            "evidence_collected": ["inspect:redis-session", "query_metrics", "query_logs", "run_health_check:redis-session"],
            "mitigations_applied": ["restart_service:redis-session"],
            "artifact_status": {"rca": True, "postmortem": True, "runbook_attached": True},
            "ignored_alert_ticks": 0,
        },
    }
    assert task_4.grade([], final_state) >= 0.9


def test_task_5_state_based_grader_requires_kafka_and_zone_recovery():
    final_state = {
        "services": {"kafka": "healthy", "worker-service": "healthy", "notification-service": "healthy", "analytics-service": "healthy"},
        "zones": {"us-central-1a": {"status": "healthy"}},
        "metrics": {"queue_depth": 1800, "error_rate": 0.03, "p99_latency_ms": 520},
        "incident": {
            "evidence_collected": ["inspect:kafka", "inspect:worker-service", "query_metrics", "query_logs", "query_traces", "query_topology"],
            "mitigations_applied": ["restart_service:kafka", "clear_queue:worker-service", "scale:worker-service", "drain_zone:us-central-1a", "restore_zone:us-central-1a"],
            "artifact_status": {"rca": True, "postmortem": True, "runbook_attached": True},
            "ignored_alert_ticks": 0,
        },
    }
    assert task_5.grade([], final_state) >= 0.9


def test_task_6_state_based_grader_requires_mesh_and_edge_recovery():
    final_state = {
        "services": {"service-mesh": "healthy", "api-gateway": "healthy", "frontend-web": "healthy"},
        "zones": {"us-east-1a": {"status": "healthy"}},
        "metrics": {"p99_latency_ms": 480, "error_rate": 0.02},
        "sla": {"verified": True, "current_status": "CURRENTLY_MET"},
        "incident": {
            "evidence_collected": ["query_metrics", "query_traces", "query_deploy", "query_topology", "inspect:service-mesh"],
            "mitigations_applied": ["rollback_deploy:service-mesh", "failover_zone:us-east-1a", "rebalance_traffic:api-gateway"],
            "artifact_status": {"rca": True, "postmortem": True, "runbook_attached": True},
            "ignored_alert_ticks": 0,
        },
    }
    assert task_6.grade([], final_state) >= 0.9


def test_task_7_replica_lag_grader_uses_dynamic_replica_targets():
    final_state = {
        "services": {"postgres-replica": "healthy", "profile-service": "healthy", "report-service": "healthy"},
        "zones": {"us-east-1b": {"status": "healthy"}},
        "metrics": {"queue_depth": 1800, "error_rate": 0.03, "p99_latency_ms": 520, "availability": 99.6},
        "sla": {"verified": True, "current_status": "CURRENTLY_MET"},
        "incident": {
            "evidence_collected": ["inspect:postgres-replica", "query_metrics", "query_logs", "query_traces", "run_health_check:postgres-replica"],
            "mitigations_applied": ["restart_service:postgres-replica", "restore_zone:us-east-1b"],
            "artifact_status": {"rca": True, "postmortem": True, "runbook_attached": True},
            "ignored_alert_ticks": 0,
        },
    }
    assert task_7.grade([], final_state) >= 0.9


def test_task_8_cache_stampede_grader_requires_cache_and_scale_recovery():
    final_state = {
        "services": {"redis-cache": "healthy", "cart-service": "healthy", "frontend-web": "healthy"},
        "metrics": {"queue_depth": 2200, "error_rate": 0.03, "p99_latency_ms": 560, "availability": 99.5},
        "sla": {"verified": True, "current_status": "CURRENTLY_MET"},
        "incident": {
            "evidence_collected": ["inspect:redis-cache", "inspect:cart-service", "query_metrics", "query_logs", "query_traces"],
            "mitigations_applied": ["restart_service:redis-cache", "scale:cart-service", "tune_autoscaling:cart-service"],
            "artifact_status": {"rca": True, "postmortem": True, "runbook_attached": True},
            "ignored_alert_ticks": 0,
        },
    }
    assert task_8.grade([], final_state) >= 0.9


def test_task_9_partial_rollback_grader_requires_restart_after_rollback():
    final_state = {
        "services": {"frontend-web": "healthy", "api-gateway": "healthy"},
        "metrics": {"error_rate": 0.03, "p99_latency_ms": 520, "availability": 99.45},
        "sla": {"verified": True, "current_status": "CURRENTLY_MET"},
        "incident": {
            "evidence_collected": ["inspect:frontend-web", "query_metrics", "query_logs", "query_traces", "query_deploy"],
            "mitigations_applied": ["rollback_deploy:frontend-web", "restart_service:frontend-web", "rebalance_traffic:api-gateway"],
            "artifact_status": {"rca": True, "postmortem": True, "runbook_attached": True},
            "ignored_alert_ticks": 0,
        },
    }
    assert task_9.grade([], final_state) >= 0.9


def test_task_10_dns_outage_grader_requires_dns_and_zone_mitigation():
    final_state = {
        "services": {"dns-control-plane": "healthy", "edge-proxy": "healthy", "api-gateway": "healthy"},
        "zones": {"us-east-1a": {"status": "healthy"}},
        "metrics": {"error_rate": 0.03, "p99_latency_ms": 540, "availability": 99.35},
        "sla": {"verified": True, "current_status": "CURRENTLY_MET"},
        "incident": {
            "evidence_collected": ["inspect:dns-control-plane", "inspect:edge-proxy", "query_metrics", "query_logs", "query_topology"],
            "mitigations_applied": ["restart_service:dns-control-plane", "failover_zone:us-east-1a", "rebalance_traffic:api-gateway"],
            "artifact_status": {"rca": True, "postmortem": True, "runbook_attached": True},
            "ignored_alert_ticks": 0,
        },
    }
    assert task_10.grade([], final_state) >= 0.9


def test_task_11_region_failure_grader_supports_multi_zone_targets():
    final_state = {
        "services": {"api-gateway": "healthy", "frontend-web": "healthy", "order-service": "healthy", "payment-service": "healthy"},
        "zones": {"us-east-1a": {"status": "healthy"}, "us-east-1b": {"status": "healthy"}},
        "metrics": {"error_rate": 0.03, "p99_latency_ms": 620, "availability": 99.25},
        "sla": {"verified": True, "current_status": "CURRENTLY_MET"},
        "incident": {
            "evidence_collected": ["inspect:api-gateway", "query_metrics", "query_logs", "query_traces", "query_topology"],
            "mitigations_applied": ["failover_zone:us-east-1a", "failover_zone:us-east-1b", "restore_zone:us-east-1a", "restore_zone:us-east-1b", "rebalance_traffic:api-gateway"],
            "artifact_status": {"rca": True, "postmortem": True, "runbook_attached": True},
            "ignored_alert_ticks": 0,
        },
    }
    assert task_11.grade([], final_state) >= 0.9


def test_graders_are_deterministic_and_clamped():
    final_state = {
        "services": {"postgres-primary": "healthy", "auth-service": "healthy", "billing-service": "healthy"},
        "metrics": {"error_rate": 0.03, "p99_latency_ms": 320},
        "sla": {"verified": True, "current_status": "CURRENTLY_MET"},
        "incident": {
            "evidence_collected": ["inspect:postgres-primary", "query_metrics", "query_logs", "run_health_check:postgres-primary"],
            "mitigations_applied": ["restart_service:postgres-primary"],
            "artifact_status": {"rca": True, "postmortem": True, "runbook_attached": True},
            "unnecessary_restarts": [],
            "ignored_alert_ticks": 0,
        },
    }
    history = [{"action": "restart_service", "target": "postgres-primary"}]
    first = task_1.grade(history, final_state)
    second = task_1.grade(history, final_state)
    assert first == second
    assert 0.0 <= first <= 1.0


def test_dynamic_variant_grading_uses_incident_targets_not_hardcoded_defaults():
    final_state = {
        "services": {"worker-service": "healthy", "scheduler-service": "healthy", "notification-service": "healthy"},
        "zones": {"us-east-1a": {"status": "healthy"}},
        "metrics": {"queue_depth": 1200, "availability": 99.95, "error_rate": 0.02, "p99_latency_ms": 280},
        "incident": {
            "service_targets": ["worker-service"],
            "zone_targets": ["us-east-1a"],
            "required_evidence": ["inspect:worker-service", "query_metrics", "query_logs", "query_traces", "query_topology"],
            "required_mitigations": ["clear_queue:worker-service", "scale:worker-service", "tune_autoscaling:worker-service", "drain_zone:us-east-1a", "restore_zone:us-east-1a"],
            "evidence_collected": ["inspect:worker-service", "query_metrics", "query_logs", "query_traces", "query_topology"],
            "mitigations_applied": ["clear_queue:worker-service", "scale:worker-service", "tune_autoscaling:worker-service", "drain_zone:us-east-1a", "restore_zone:us-east-1a"],
            "artifact_status": {"runbook_attached": True, "rca": True, "postmortem": True},
            "production_actions_completed": ["verify_sla", "attach_runbook", "run_rca", "generate_postmortem"],
            "ignored_alert_ticks": 0,
        },
        "sla": {"verified": True, "current_status": "CURRENTLY_MET"},
    }
    assert task_2.grade([], final_state) >= 0.95


def test_manager_communications_contribute_partial_credit():
    final_state = {
        "services": {"postgres-primary": "healthy", "auth-service": "healthy", "billing-service": "healthy"},
        "metrics": {"error_rate": 0.03, "p99_latency_ms": 320, "availability": 99.95},
        "sla": {"verified": True, "current_status": "CURRENTLY_MET"},
        "incident": {
            "service_targets": ["postgres-primary"],
            "required_evidence": ["inspect:postgres-primary", "query_metrics", "query_logs", "run_health_check:postgres-primary"],
            "required_mitigations": ["restart_service:postgres-primary"],
            "evidence_collected": ["inspect:postgres-primary", "query_metrics", "query_logs", "run_health_check:postgres-primary"],
            "mitigations_applied": ["restart_service:postgres-primary"],
            "artifact_status": {"runbook_attached": True, "rca": True, "postmortem": True},
            "production_actions_completed": ["verify_sla", "attach_runbook", "run_rca", "generate_postmortem", "notify"],
            "needs_manager_comms": True,
            "notified": ["stakeholders"],
            "escalated_to": [],
            "status_page_updated": False,
            "ignored_alert_ticks": 0,
        },
    }
    score = task_1.grade([{"action": "notify"}], final_state)
    assert 0.7 <= score < 1.0


def test_chaos_grader_supports_dynamic_targets_and_partial_credit():
    final_state = {
        "services": {
            "frontend-web": "healthy",
            "order-service": "healthy",
            "postgres-primary": "healthy",
            "search-service": "degraded",
        },
        "zones": {"us-east-1b": {"status": "healthy"}, "us-east-1c": {"status": "healthy"}},
        "metrics": {"queue_depth": 4200, "availability": 99.1, "error_rate": 0.05, "p99_latency_ms": 640},
        "sla": {"verified": True, "current_status": "CURRENTLY_MET"},
        "incident": {
            "service_targets": ["frontend-web", "order-service", "postgres-primary", "search-service"],
            "zone_targets": ["us-east-1b", "us-east-1c"],
            "required_evidence": ["inspect:frontend-web", "inspect:order-service", "query_metrics", "query_logs"],
            "required_mitigations": ["restart_service:order-service", "restart_service:search-service", "failover_zone:us-east-1b"],
            "evidence_collected": ["inspect:frontend-web", "inspect:order-service", "query_metrics"],
            "mitigations_applied": ["restart_service:order-service", "failover_zone:us-east-1b"],
            "artifact_status": {"runbook_attached": True, "rca": True, "postmortem": False},
            "production_actions_completed": ["verify_sla", "attach_runbook", "run_rca", "notify", "update_status_page"],
            "needs_manager_comms": True,
            "notified": ["stakeholders"],
            "status_page_updated": True,
            "escalated_to": [],
            "ignored_alert_ticks": 1,
        },
    }
    score = chaos.grade([], final_state)
    assert 0.3 <= score <= 0.9
