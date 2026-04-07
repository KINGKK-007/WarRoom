from environment.env import DevOpsWarRoomEnv
from environment.models import Scenario, ServiceState
from graders.task_1 import grade as grade_task_1
from graders.task_2 import grade as grade_task_2
from graders.task_3 import grade as grade_task_3


def _run(env: DevOpsWarRoomEnv, command: str):
    return env.step({"action_type": "raw_command", "params": {"command": command}})


def test_expanded_environment_supports_30_plus_services():
    env = DevOpsWarRoomEnv()
    obs = env.reset(Scenario.EASY)
    assert len(obs.services) >= 30


def test_task_1_playbook_scores_well():
    env = DevOpsWarRoomEnv()
    env.reset(Scenario.EASY)
    for command in [
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
    ]:
        _run(env, command)
    score = grade_task_1(env.action_history, env.state)
    assert env.state["services"]["postgres-primary"] == ServiceState.healthy
    assert score >= 0.9


def test_task_2_playbook_scores_well():
    env = DevOpsWarRoomEnv()
    env.reset(Scenario.MEDIUM)
    for command in [
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
    ]:
        _run(env, command)
    score = grade_task_2(env.action_history, env.state)
    assert env.state["services"]["worker-service"] == ServiceState.healthy
    assert score >= 0.9


def test_task_3_playbook_scores_well():
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
    ]:
        _run(env, command)
    score = grade_task_3(env.action_history, env.state)
    assert env.state["services"]["api-gateway"] == ServiceState.healthy
    assert score >= 0.9
