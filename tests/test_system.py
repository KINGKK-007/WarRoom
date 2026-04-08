import time

from fastapi.testclient import TestClient

from environment.actions import ACTION_CATALOG
from environment.env import DevOpsWarRoomEnv
from environment.models import Role, Scenario, ServiceState
from environment.server import app


def _run(env: DevOpsWarRoomEnv, command: str):
    return env.step({"action_type": "raw_command", "params": {"command": command}})


client = TestClient(app)


EASY_PLAYBOOK = [
    "acknowledge alert",
    "inspect postgres-primary",
    "query metrics",
    "query logs",
    "run health check postgres-primary",
    "restart service postgres-primary",
    "verify sla",
    "run rca",
    "switch role Manager",
    "attach runbook",
    "notify stakeholders",
    "update status page database",
    "switch role Dev",
    "generate postmortem",
]

MEDIUM_PLAYBOOK = [
    "acknowledge alert",
    "inspect worker-service",
    "query metrics",
    "query logs",
    "query traces",
    "query topology",
    "clear queue worker-service",
    "scale worker-service 3",
    "tune autoscaling worker-service",
    "drain zone us-east-1b",
    "restore zone us-east-1b",
    "verify sla",
    "run rca",
    "switch role Manager",
    "attach runbook",
    "notify stakeholders",
    "update status page workers",
    "switch role Dev",
    "generate postmortem",
]

HARD_PLAYBOOK = [
    "acknowledge alert",
    "query metrics",
    "query traces",
    "query topology",
    "inspect api-gateway",
    "switch role Dev",
    "query deploy history",
    "rollback api-gateway v3.2.0",
    "switch role SRE",
    "failover zone us-east-1c",
    "rebalance traffic api-gateway",
    "restore zone us-east-1c",
    "verify sla",
    "run rca",
    "switch role Manager",
    "attach runbook",
    "notify stakeholders",
    "update status page api-gateway",
    "switch role Dev",
    "generate postmortem",
]


class TestProductionTopology:
    def test_topology_has_30_plus_services(self):
        env = DevOpsWarRoomEnv()
        obs = env.reset(Scenario.EASY)
        assert len(obs.services) >= 30
        assert len(obs.service_distribution) >= 30
        assert len(obs.zone_health) == 4

    def test_action_space_has_20_plus_meaningful_actions(self):
        assert len(ACTION_CATALOG) >= 20
        assert "run_rca" in ACTION_CATALOG
        assert "generate_postmortem" in ACTION_CATALOG
        assert "failover_zone" in ACTION_CATALOG

    def test_chaos_scenario_is_generated(self):
        env = DevOpsWarRoomEnv()
        obs = env.reset(Scenario.CHAOS)
        assert len(obs.services) >= 30
        assert len(env.state["incident"]["root_causes"]) >= 5
        assert env.state["incident"]["severity"] == "sev0"

    def test_seeded_variant_is_deterministic(self):
        env_a = DevOpsWarRoomEnv()
        env_b = DevOpsWarRoomEnv()
        env_a.reset(Scenario.MEDIUM, seed=123)
        env_b.reset(Scenario.MEDIUM, seed=123)
        assert env_a.state["incident"]["variant"] == env_b.state["incident"]["variant"]
        assert env_a.state["incident"]["required_mitigations"] == env_b.state["incident"]["required_mitigations"]

    def test_state_returns_isolated_snapshot(self):
        env = DevOpsWarRoomEnv()
        snapshot = env.state
        snapshot["metrics"]["error_rate"] = 0.99
        assert env.state["metrics"]["error_rate"] != 0.99

    def test_reset_accepts_task_alias_string(self):
        env = DevOpsWarRoomEnv()
        obs = env.reset("task_1")
        assert obs.incident_summary["task_id"] == Scenario.EASY.value

    def test_api_reset_accepts_manifest_task_aliases(self):
        response = client.post("/reset", json={"task_id": "task_4", "seed": 123})
        assert response.status_code == 200
        payload = response.json()
        assert payload["observation"]["incident_summary"]["task_id"] == Scenario.EasyRedis.value
        assert payload["session_id"].startswith("sess-")

    def test_api_reset_accepts_new_advanced_task_aliases(self):
        response = client.post("/reset", json={"task_id": "task_10", "seed": 123})
        assert response.status_code == 200
        payload = response.json()
        assert payload["observation"]["incident_summary"]["task_id"] == Scenario.HardDNS.value
        assert payload["session_id"].startswith("sess-")

    def test_api_step_reward_is_bounded(self):
        reset_payload = client.post("/reset", json={"task_id": "task_1"}).json()
        response = client.post(
            "/step",
            json={
                "session_id": reset_payload["session_id"],
                "action_type": "raw_command",
                "params": {"command": "unknown command"},
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert 0.0 <= payload["reward"]["value"] <= 1.0

    def test_api_state_is_session_scoped_and_isolated(self):
        first = client.post("/reset", json={"task_id": "task_1", "seed": 123}).json()
        second = client.post("/reset", json={"task_id": "task_1", "seed": 123}).json()

        client.post(
            "/step",
            json={
                "session_id": first["session_id"],
                "action_type": "raw_command",
                "params": {"command": "query metrics"},
            },
        )

        first_state = client.get("/state", params={"session_id": first["session_id"]}).json()["state"]
        second_state = client.get("/state", params={"session_id": second["session_id"]}).json()["state"]
        assert first_state["metrics_history"][-1]["label"] == "query_metrics"
        assert second_state["metrics_history"][-1]["label"] == "reset"


class TestRoleVisibility:
    def test_sre_sees_metrics_logs_traces(self):
        env = DevOpsWarRoomEnv(role=Role.SRE)
        obs = env.reset(Scenario.HARD)
        assert obs.metrics is not None
        assert obs.logs is not None
        assert obs.traces is not None
        assert obs.deployment_history is None

    def test_dev_sees_deploy_history(self):
        env = DevOpsWarRoomEnv()
        env.reset(Scenario.HARD)
        obs, _, _, _ = _run(env, "switch role dev")
        assert obs.metrics is None
        assert obs.deployment_history is not None
        assert obs.code_diffs is not None

    def test_manager_sees_sla_and_affected_users(self):
        env = DevOpsWarRoomEnv()
        env.reset(Scenario.HARD)
        obs, _, _, _ = _run(env, "switch role Manager")
        assert obs.sla_status is not None
        assert obs.estimated_affected_users is not None
        assert obs.metrics is None

    def test_observation_exposes_incident_progress_and_suggestions(self):
        env = DevOpsWarRoomEnv()
        obs = env.reset(Scenario.EASY)
        assert "artifact_status" in obs.incident_summary
        assert isinstance(obs.suggested_actions, list)
        assert len(obs.suggested_actions) > 0


class TestIncidentMechanics:
    def test_easy_recovery_flow_restores_primary_database(self):
        env = DevOpsWarRoomEnv()
        env.reset(Scenario.EASY)
        for command in EASY_PLAYBOOK:
            _run(env, command)
        assert env.state["services"]["postgres-primary"] == ServiceState.healthy
        assert env.state["incident"]["artifact_status"]["rca"] is True
        assert env.state["incident"]["artifact_status"]["postmortem"] is True
        assert env.state["incident"]["artifact_status"]["runbook_attached"] is True
        assert env.state["incident"]["recovery_tick"] is not None

    def test_medium_dense_rewards_and_queue_reduction(self):
        env = DevOpsWarRoomEnv()
        env.reset(Scenario.MEDIUM)
        _, reward, _, _ = _run(env, "inspect worker-service")
        assert reward.value > 0
        for command in MEDIUM_PLAYBOOK[2:]:
            _run(env, command)
        assert env.state["metrics"]["queue_depth"] <= 5000
        assert env.state["services"]["worker-service"] == ServiceState.healthy
        assert env.state["zones"]["us-east-1b"]["status"] == "healthy"

    def test_hard_multi_zone_playbook(self):
        env = DevOpsWarRoomEnv()
        env.reset(Scenario.HARD)
        for command in HARD_PLAYBOOK:
            _run(env, command)
        assert env.state["zones"]["us-east-1c"]["status"] == "healthy"
        assert env.state["metrics"]["p99_latency_ms"] <= 600
        assert env.state["services"]["api-gateway"] == ServiceState.healthy
        assert env.state["incident"]["recovery_tick"] is not None

    def test_sla_breach_can_end_episode(self):
        env = DevOpsWarRoomEnv()
        env.reset(Scenario.CHAOS)
        env._state["sla"]["breaches"] = [{"tick": 0}, {"tick": 1}, {"tick": 2}]
        assert env._is_done() is True

    def test_step_after_episode_end_returns_terminal_noop(self):
        env = DevOpsWarRoomEnv()
        env.reset(Scenario.EASY)
        env._done = True
        before = env.state
        obs, reward, done, info = _run(env, "query metrics")
        assert done is True
        assert reward.value == 0.0
        assert info["error"] == "episode_complete"
        assert obs.tick == before["metrics_history"][-1]["tick"]

    def test_rca_contains_causal_chain_and_blast_radius(self):
        env = DevOpsWarRoomEnv()
        env.reset(Scenario.HARD)
        for command in [
            "acknowledge alert",
            "query metrics",
            "query traces",
            "query topology",
            "inspect api-gateway",
            "switch role Dev",
            "query deploy history",
            "rollback api-gateway v3.2.0",
            "attach runbook",
            "switch role SRE",
            "failover zone us-east-1c",
            "rebalance traffic api-gateway",
            "restore zone us-east-1c",
            "verify sla",
            "run rca",
        ]:
            _run(env, command)
        rca = env.state["incident"]["rca"]
        assert "causal_chain" in rca
        assert "blast_radius" in rca
        assert isinstance(env.timeline(), list)


class TestValidationBenchmark:
    def test_playbook_benchmark_runs_fast_and_resolves_incidents(self):
        playbooks = {Scenario.EASY: EASY_PLAYBOOK, Scenario.MEDIUM: MEDIUM_PLAYBOOK, Scenario.HARD: HARD_PLAYBOOK}
        started = time.perf_counter()
        for scenario, commands in playbooks.items():
            env = DevOpsWarRoomEnv()
            env.reset(scenario)
            for command in commands:
                _run(env, command)
            assert env.state["incident"]["artifact_status"]["rca"] is True
            assert env.state["incident"]["artifact_status"]["postmortem"] is True
            assert env.state["incident"]["recovery_tick"] is not None
            assert env.state["metrics"]["error_rate"] <= 0.08
        assert time.perf_counter() - started < 2.0
