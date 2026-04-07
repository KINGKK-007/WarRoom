import random
import time

from environment.env import DevOpsWarRoomEnv
from environment.models import Scenario


SMART_PLAYBOOKS = {
    Scenario.EASY: [
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
    ],
    Scenario.MEDIUM: [
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
    ],
    Scenario.HARD: [
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
    ],
}


def run_random_agent(scenario: Scenario, seed: int = 7):
    env = DevOpsWarRoomEnv()
    obs = env.reset(scenario)
    rng = random.Random(seed)
    while True:
        action_name = rng.choice(obs.available_actions)
        command = action_name.replace("_", " ")
        if action_name == "switch_role":
            command = f"switch role {rng.choice(['SRE', 'Dev', 'Manager'])}"
        elif action_name in {"inspect", "restart_service", "clear_queue", "scale", "tune_autoscaling", "rebalance_traffic", "throttle_service", "isolate_service", "run_health_check"}:
            command = f"{command} {rng.choice(list(env.state['services'].keys()))}"
            if action_name == "scale":
                command += " 2"
        elif action_name in {"failover_zone", "drain_zone", "restore_zone"}:
            command = f"{command} {rng.choice(list(env.state['zones'].keys()))}"
        elif action_name == "rollback_deploy":
            command = f"rollback {rng.choice(list(env.state['services'].keys()))} v1.0.0"
        obs, _, done, _ = env.step({"action_type": "raw_command", "params": {"command": command}})
        if done:
            break
    return env


def run_smart_agent(scenario: Scenario):
    env = DevOpsWarRoomEnv()
    env.reset(scenario)
    for command in SMART_PLAYBOOKS[scenario]:
        env.step({"action_type": "raw_command", "params": {"command": command}})
    return env


def main():
    started = time.perf_counter()
    for scenario in [Scenario.EASY, Scenario.MEDIUM, Scenario.HARD]:
        smart = run_smart_agent(scenario)
        random_env = run_random_agent(scenario)
        print(
            {
                "scenario": scenario.value,
                "smart_error_rate": smart.state["metrics"]["error_rate"],
                "smart_resolved": smart.state["incident"]["recovery_tick"] is not None,
                "random_error_rate": random_env.state["metrics"]["error_rate"],
                "random_resolved": random_env.state["incident"]["recovery_tick"] is not None,
            }
        )
    print({"runtime_s": round(time.perf_counter() - started, 4)})


if __name__ == "__main__":
    main()
