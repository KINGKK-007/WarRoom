"""
inference.py — OpenEnv Baseline Agent for DevOps War Room

Reads OPENAI_API_KEY from environment variable (set as HuggingFace Space secret).
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

# ---------------------------------------------------------------------------
# System prompt — expert SRE decision protocol
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert on-call SRE responding to a live production incident.

You operate in a role-based environment. Your role controls what you can see and do.

ROLES:
- SRE: query_metrics, restart_service, inspect, scale. Sees: error_rate, cpu, memory, p99_latency_ms, requests_per_sec, services, alerts.
- Dev: rollback_deploy, query_deploy, query_logs. Sees: deployment_history, code_diffs, logs.
- Manager: escalate, notify. Sees: sla_status, affected_users.
- All roles: switch_role (costs 1 step, use sparingly).

STRICT DECISION PROTOCOL — follow this every episode:
1. Start as SRE. Run "query metrics" first. Read error_rate and p99_latency_ms.
2. Check alerts. If a CRITICAL alert names a service, that is your target.
3. If all services appear healthy but latency is high (p99 > 500ms), switch to Dev and run "query deploy history".
4. Look for a recent deployment that correlates with the incident start time.
5. If a bad deploy is identified, run "rollback [service] [version]" where version is the previous stable one.
6. If a service is DOWN or DEGRADED with a clear alert, run "restart service [name]".
7. After fixing, run "query metrics" again to confirm error_rate is dropping.
8. Never restart a service that is already healthy.
9. Never repeat the same action twice in a row.
10. Never issue more than one command per response.

OUTPUT FORMAT:
Respond with a single command string only. No explanation. No punctuation. Examples:
query metrics
restart service database
switch role dev
query deploy history
rollback api-service v2.3.0
"""


def build_prompt(obs_dict: dict) -> str:
    """Build a role-aware prompt from the observation dictionary.

    Adapts the prompt based on the current role so the LLM sees
    role-specific information (deploy history for Dev, SLA for Manager, etc.).
    """
    role = obs_dict.get("current_role", "SRE")
    tick = obs_dict.get("tick", 0)
    steps_remaining = obs_dict.get("steps_remaining", "?")

    prompt = (
        f"Current role: {role} | Tick: {tick} | Steps remaining: {steps_remaining}\n\n"
    )

    # Services — always visible
    services = obs_dict.get("services", {})
    prompt += "SERVICE STATUS:\n"
    for svc, status in services.items():
        svc_str = str(status)
        if svc_str == "healthy":
            indicator = "✅"
        elif svc_str == "degraded":
            indicator = "⚠️"
        else:
            indicator = "❌"
        prompt += f"  {indicator} {svc}: {svc_str}\n"
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
    else:
        prompt += "ACTIVE ALERTS: None\n\n"

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
        "\nAvailable actions for your current role:\n"
    )
    if role == "SRE":
        prompt += (
            "  restart service {name}    — restart a broken service\n"
            "  inspect {target}          — inspect a service\n"
            "  query metrics             — view current metrics\n"
            "  scale {target}            — scale a service\n"
            "  switch role {dev|manager} — switch role (costs 1 step)\n"
        )
    elif role == "Dev":
        prompt += (
            "  rollback deploy {name} {version} — rollback a deployment\n"
            "  query deploy history      — view deploy history\n"
            "  query logs                — view recent logs\n"
            "  switch role {sre|manager} — switch role (costs 1 step)\n"
        )
    elif role == "Manager":
        prompt += (
            "  escalate {target}         — escalate incident\n"
            "  notify {target}           — notify stakeholders\n"
            "  switch role {sre|dev}     — switch role (costs 1 step)\n"
        )

    prompt += "\nRespond with ONLY the exact command to execute. No explanation."

    return prompt


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def main():
    # Read required environment variables
    api_base_url = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable not set. "
            "Add it as a HuggingFace Space secret at: "
            "https://huggingface.co/spaces/coolalien35/warroom-deploy/settings"
        )

    # Initialize OpenAI client with timeout protection
    client = OpenAI(
        base_url=api_base_url,
        api_key=api_key,
        timeout=LLM_CALL_TIMEOUT_SECONDS,
        max_retries=2,
    )

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

            # --- Decide action via LLM ---
            action_text = "query metrics"  # safe fallback
            try:
                obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs.dict()
                prompt = build_prompt(obs_dict)

                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=64,
                    temperature=0.0,
                )
                raw = response.choices[0].message.content.strip()
                # Clean up any markdown formatting the LLM might add
                action_text = raw.strip("`").strip('"').strip("'").split("\n")[0].strip()
            except Exception as e:
                print(f"[LLM ERROR] {e}", flush=True)
                # Fallback on any LLM error (timeout, rate limit, auth, etc.)
                action_text = "query metrics"

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
