from graders import task_1, task_2, task_3


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
