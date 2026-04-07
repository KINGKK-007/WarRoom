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
