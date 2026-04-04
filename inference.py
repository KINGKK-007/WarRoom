"""
inference.py — OpenEnv Baseline Agent for DevOps War Room

Reads API_BASE_URL, MODEL_NAME, HF_TOKEN from environment variables.
Uses the OpenAI client to query an LLM for action decisions.
Logs structured JSON in the exact START / STEP / END format required
by the OpenEnv Phase 1 automated validator.

Runtime constraint: must complete all 3 tasks within 20 minutes.
"""

import os
import json
import time
import signal
from openai import OpenAI
from environment.env import DevOpsWarRoomEnv
from environment.models import Scenario, Action
from graders import task_1, task_2, task_3


# ---------------------------------------------------------------------------
# Global runtime guard — hard kill at 19 minutes to stay under 20-min limit
# ---------------------------------------------------------------------------
GLOBAL_TIMEOUT_SECONDS = 19 * 60  # 19 minutes (1 min buffer)
PER_TASK_TIMEOUT_SECONDS = 5 * 60  # 5 minutes per task
LLM_CALL_TIMEOUT_SECONDS = 30      # individual API call timeout
MAX_STEPS_PER_TASK = 15


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Global 20-minute runtime limit approaching — aborting.")


# Install global timeout (Unix only; harmless no-op on Windows)
try:
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(GLOBAL_TIMEOUT_SECONDS)
except (AttributeError, OSError):
    pass  # Windows or restricted environment — skip alarm


# ---------------------------------------------------------------------------
# Structured JSON logging — exact format required by OpenEnv spec
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str):
    print(json.dumps({
        "type": "START",
        "task": task,
        "env": env,
        "model": model
    }), flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error):
    print(json.dumps({
        "type": "STEP",
        "step": step,
        "action": action,
        "reward": reward,
        "done": done,
        "error": error
    }), flush=True)


def log_end(success: bool, steps: int, score: float, rewards: list):
    print(json.dumps({
        "type": "END",
        "success": success,
        "steps": steps,
        "score": score,
        "rewards": rewards
    }), flush=True)


# ---------------------------------------------------------------------------
# Prompt engineering — role-adaptive observation formatting
# ---------------------------------------------------------------------------

def build_prompt(obs_dict: dict) -> str:
    """Build a role-aware prompt from the observation dictionary.

    Adapts the prompt based on the current role so the LLM sees
    role-specific information (deploy history for Dev, SLA for Manager, etc.).
    """
    role = obs_dict.get("current_role", "SRE")
    tick = obs_dict.get("tick", 0)

    prompt = (
        f"You are an expert AI SRE/DevOps incident responder.\n"
        f"Current role: {role} | Tick: {tick}\n\n"
    )

    # Services — always visible
    services = obs_dict.get("services", {})
    prompt += "SERVICE STATUS:\n"
    for svc, status in services.items():
        indicator = "✅" if str(status) == "up" else ("⚠️" if str(status) == "degraded" else "❌")
        prompt += f"  {indicator} {svc}: {status}\n"
    prompt += "\n"

    # Alerts — always visible
    alerts = obs_dict.get("alerts", [])
    if alerts:
        prompt += "ACTIVE ALERTS:\n"
        for alert in alerts:
            if isinstance(alert, dict):
                prompt += f"  [{alert.get('severity', 'UNKNOWN')}] {alert.get('message', '')}\n"
            else:
                prompt += f"  {alert}\n"
        prompt += "\n"

    # Metrics — SRE role
    metrics = obs_dict.get("metrics")
    if metrics:
        prompt += "METRICS:\n"
        for k, v in metrics.items():
            prompt += f"  {k}: {v}\n"
        prompt += "\n"

    # Logs — SRE and Dev roles
    logs = obs_dict.get("logs")
    if logs:
        prompt += "RECENT LOGS:\n"
        for line in logs[-10:]:
            prompt += f"  {line}\n"
        prompt += "\n"

    # Deployment history — Dev role
    deploy_history = obs_dict.get("deployment_history")
    if deploy_history:
        prompt += "DEPLOYMENT HISTORY:\n"
        for entry in deploy_history:
            prompt += f"  {entry}\n"
        prompt += "\n"

    # Code diffs — Dev role
    code_diffs = obs_dict.get("code_diffs")
    if code_diffs:
        prompt += "CODE DIFFS:\n"
        for diff in code_diffs:
            prompt += f"  {diff}\n"
        prompt += "\n"

    # SLA status — Manager role
    sla = obs_dict.get("sla_status")
    if sla:
        prompt += f"SLA STATUS: {sla}\n"

    affected = obs_dict.get("estimated_affected_users")
    if affected is not None:
        prompt += f"ESTIMATED AFFECTED USERS: {affected}\n"

    prompt += (
        "\nAvailable actions:\n"
        "  restart service {name}    — (SRE) restart a broken service\n"
        "  rollback deploy {name} {version} — (Dev) rollback a deployment\n"
        "  switch role {sre|dev|manager}    — switch observation perspective\n"
        "  inspect {target}          — (SRE) inspect a service\n"
        "  query metrics             — (SRE) view current metrics\n"
        "  query deploy              — (Dev) view deploy history\n"
        "  query logs                — (Dev) view recent logs\n"
        "  scale {target}            — (SRE) scale a service\n"
        "  escalate {target}         — (Manager) escalate incident\n"
        "  notify {target}           — (Manager) notify stakeholders\n"
        "\nRespond with ONLY the exact command to execute. No explanation."
    )

    return prompt


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def main():
    # Read required environment variables
    api_base_url = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    model_name = os.environ.get("MODEL_NAME", "gpt-4-turbo")
    api_key = os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY", "dummy-key"))

    # Initialize OpenAI client with timeout protection
    client = None
    try:
        client = OpenAI(
            base_url=api_base_url,
            api_key=api_key,
            timeout=LLM_CALL_TIMEOUT_SECONDS,
            max_retries=1,
        )
    except Exception as e:
        print(f"Warning: OpenAI client init failed: {e}", flush=True)

    # Initialize environment
    env = DevOpsWarRoomEnv()

    # Task definitions: (scenario_enum, grader_function)
    tasks = [
        (Scenario.EASY,   task_1.grade),
        (Scenario.MEDIUM, task_2.grade),
        (Scenario.HARD,   task_3.grade),
    ]

    for task_enum, grader_func in tasks:
        task_id = task_enum.value
        task_start_time = time.time()

        log_start(task=task_id, env="devops-warroom", model=model_name)

        # Reset environment for this task
        obs = env.reset(task_enum)
        done = False
        step_count = 0
        action_history = []
        reward_history = []

        while not done and step_count < MAX_STEPS_PER_TASK:
            step_count += 1

            # --- Per-task timeout check ---
            elapsed = time.time() - task_start_time
            if elapsed > PER_TASK_TIMEOUT_SECONDS:
                log_step(
                    step=step_count,
                    action="<timeout>",
                    reward=0.0,
                    done=True,
                    error="Per-task timeout reached"
                )
                break

            # --- Decide action via LLM or fallback ---
            action_text = "query metrics"  # safe fallback
            if client:
                try:
                    obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs.dict()
                    prompt = build_prompt(obs_dict)

                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are resolving a production incident in a DevOps war room. "
                                    "Respond with exactly one command. No explanation, no markdown."
                                ),
                            },
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=64,
                        temperature=0.0,
                    )
                    raw = response.choices[0].message.content.strip()
                    # Clean up any markdown formatting the LLM might add
                    action_text = raw.strip("`").strip('"').strip("'").split("\n")[0].strip()
                except Exception:
                    # Fallback on any LLM error (timeout, rate limit, auth, etc.)
                    action_text = "switch role sre"

            # --- Execute action in environment ---
            action_payload = Action(action_type="raw_command", target=action_text)
            obs, reward, done_env, info_env = env.step(action_payload)
            done = done_env

            action_history.append(action_text)
            reward_history.append(reward.value)

            log_step(
                step=step_count,
                action=action_text,
                reward=reward.value,
                done=done_env,
                error=info_env.get("error"),
            )

        # --- Grade the completed trajectory ---
        score = grader_func(action_history, env.state)
        success = score > 0.0

        log_end(
            success=success,
            steps=step_count,
            score=round(score, 4),
            rewards=reward_history,
        )

    # Cancel the global alarm if we finish in time
    try:
        signal.alarm(0)
    except (AttributeError, OSError):
        pass


if __name__ == "__main__":
    main()
