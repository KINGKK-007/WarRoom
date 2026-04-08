from environment.env import DevOpsWarRoomEnv
from environment.models import Scenario


def test_dense_rewards_cover_diagnostics_and_production_artifacts():
    env = DevOpsWarRoomEnv()
    env.reset(Scenario.HARD)

    _, reward_metrics, _, _ = env.step({"action_type": "raw_command", "params": {"command": "query metrics"}})
    _, reward_traces, _, _ = env.step({"action_type": "raw_command", "params": {"command": "query traces"}})
    _, _, _, _ = env.step({"action_type": "raw_command", "params": {"command": "query topology"}})
    _, _, _, _ = env.step({"action_type": "raw_command", "params": {"command": "inspect api-gateway"}})
    env.step({"action_type": "raw_command", "params": {"command": "switch role dev"}})
    _, reward_deploy, _, _ = env.step({"action_type": "raw_command", "params": {"command": "query deploy history"}})

    assert reward_metrics.value > 0
    assert reward_traces.value > 0
    assert reward_deploy.value > 0
    assert "query_metrics" in env.state["incident"]["evidence_collected"]
    assert "query_traces" in env.state["incident"]["evidence_collected"]
    assert "query_deploy" in env.state["incident"]["evidence_collected"]


def test_reward_function_gives_progress_credit_for_real_recovery():
    env = DevOpsWarRoomEnv()
    env.reset(Scenario.EASY)
    env.step({"action_type": "raw_command", "params": {"command": "inspect postgres-primary"}})
    env.step({"action_type": "raw_command", "params": {"command": "query metrics"}})
    env.step({"action_type": "raw_command", "params": {"command": "query logs"}})
    env.step({"action_type": "raw_command", "params": {"command": "run health check postgres-primary"}})
    _, reward_restart, _, _ = env.step({"action_type": "raw_command", "params": {"command": "restart service postgres-primary"}})

    assert reward_restart.value > 0.2
    assert env.state["metrics"]["error_rate"] < 0.18


def test_reward_function_penalizes_redundant_bad_actions():
    env = DevOpsWarRoomEnv()
    env.reset(Scenario.EASY)
    env.step({"action_type": "raw_command", "params": {"command": "inspect postgres-primary"}})
    env.step({"action_type": "raw_command", "params": {"command": "query metrics"}})
    env.step({"action_type": "raw_command", "params": {"command": "query logs"}})
    env.step({"action_type": "raw_command", "params": {"command": "run health check postgres-primary"}})
    _, reward_first, _, _ = env.step({"action_type": "raw_command", "params": {"command": "restart service postgres-primary"}})
    _, reward_repeat, _, _ = env.step({"action_type": "raw_command", "params": {"command": "restart service postgres-primary"}})

    assert 0.0 <= reward_repeat.value <= 1.0
    assert reward_repeat.value < reward_first.value


def test_all_step_rewards_are_bounded_to_openenv_range():
    env = DevOpsWarRoomEnv()
    env.reset(Scenario.EASY)

    for command in [
        "switch role Manager",
        "restart service frontend-web",
        "query metrics",
        "restart service postgres-primary",
        "unknown command",
    ]:
        _, reward, _, _ = env.step({"action_type": "raw_command", "params": {"command": command}})
        assert 0.0 <= reward.value <= 1.0


def test_restart_recovery_is_delayed_under_dependency_pressure():
    env = DevOpsWarRoomEnv()
    env.reset(Scenario.MEDIUM, seed=123)

    env.step({"action_type": "raw_command", "params": {"command": "restart service scheduler-service"}})
    assert env.state["services"]["scheduler-service"] in {"degraded", "down"} or getattr(env.state["services"]["scheduler-service"], "value", "") in {"degraded", "down"}

    env.step({"action_type": "raw_command", "params": {"command": "query metrics"}})
    env.step({"action_type": "raw_command", "params": {"command": "query metrics"}})
    assert getattr(env.state["services"]["scheduler-service"], "value", env.state["services"]["scheduler-service"]) in {"degraded", "healthy"}


def test_clear_queue_is_partial_when_kafka_is_impaired():
    env = DevOpsWarRoomEnv()
    env.reset(Scenario.MediumKafka, seed=123)
    before = env.state["service_details"]["worker-service"]["queue_depth"]
    env.step({"action_type": "raw_command", "params": {"command": "clear queue worker-service"}})
    after = env.state["service_details"]["worker-service"]["queue_depth"]
    assert after > 0
    assert after > before * 0.2


def test_dependency_pressure_expands_blast_radius():
    env = DevOpsWarRoomEnv()
    env.reset(Scenario.HARD, seed=123)
    env.step({"action_type": "raw_command", "params": {"command": "query metrics"}})
    env.step({"action_type": "raw_command", "params": {"command": "query metrics"}})
    blast_radius = env.state["incident"]["blast_radius"]
    assert "frontend-web" in blast_radius or "mobile-bff" in blast_radius or "api-gateway" in blast_radius
