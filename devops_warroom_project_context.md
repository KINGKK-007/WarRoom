# DevOps War Room — Complete Project Context
> OpenEnv Hackathon | Meta × PyTorch × HuggingFace × Scaler School of Technology
> Team: **BholeChature** | Pulkit Pandey + Kanav Kumar
> Deadline: **8 April 2026, 11:59 PM IST**
> Finale: 25–26 April 2026 (48-hour in-person, Bangalore)

---

## Table of Contents
1. [Hackathon Overview](#hackathon-overview)
2. [What We Are Building](#what-we-are-building)
3. [Why This Idea Wins](#why-this-idea-wins)
4. [Project Concept — Deep Dive](#project-concept--deep-dive)
5. [The Three Tasks](#the-three-tasks)
6. [Environment Design](#environment-design)
7. [Reward Function Design](#reward-function-design)
8. [Multi-Role System](#multi-role-system)
9. [Technical Architecture](#technical-architecture)
10. [OpenEnv Spec Requirements](#openenv-spec-requirements)
11. [Evaluation Criteria & How We Score](#evaluation-criteria--how-we-score)
12. [Disqualification Risks & Mitigations](#disqualification-risks--mitigations)
13. [File Structure](#file-structure)
14. [Implementation Plan](#implementation-plan)
15. [Key Constraints](#key-constraints)

---

## Hackathon Overview

**Name:** Meta PyTorch OpenEnv Hackathon × Scaler School of Technology  
**Prize Pool:** $30,000  
**Top Prize:** Direct interview opportunity at Meta & HuggingFace AI teams  
**Scale:** Expected 70,000+ developers nationally (India)  

**What participants build:** A complete OpenEnv reinforcement learning environment. Not a demo, not a prototype — real, deployable infrastructure that an AI agent can learn from.

**What OpenEnv is:** An open-source framework by Meta & HuggingFace for creating standardized, isolated, reusable RL environments. It uses a Gymnasium-style API (`step()` / `reset()` / `state()`), containerized execution via Docker, and publishes to HuggingFace Spaces. Think of it as the "universal language" for AI training environments.

**Judging Phases:**
- **Phase 1 (Automated):** Pass/fail gate. HF Space deploys, OpenEnv spec valid, Dockerfile builds, baseline script runs, 3+ graded tasks.
- **Phase 2 (Agentic):** Scored. A standard LLM agent (Nemotron 3 Super) is run against all environments. Score variance is checked.
- **Phase 3 (Human):** Top submissions reviewed by Meta and HuggingFace engineers for real-world utility, creativity, and exploit checks.

---

## What We Are Building

**Project Name:** DevOps War Room (working title — can be renamed)

**One-line pitch:** A living infrastructure incident simulator where an AI agent plays the role of an on-call SRE, diagnosing and resolving production incidents in a dynamically degrading environment — before the system collapses completely.

**The core idea in plain English:**
The environment simulates a production software system (multiple microservices, databases, workers). Something goes wrong — services go down, memory leaks, errors cascade. The AI agent receives alerts, metrics, and logs, and must take actions to diagnose and fix the incident. Critically, the environment is *alive* — it keeps getting worse every step the agent doesn't act correctly. Wrong actions have real consequences. The agent is racing against a deteriorating system.

**What makes it unique:**
1. The environment degrades dynamically every tick — not a static snapshot
2. The agent can switch between three roles (SRE / Dev / Manager), each seeing different information and having access to different actions
3. Wrong actions make things worse, not just "no effect"
4. Reward is granular — partial credit per correct step, not just binary pass/fail
5. Nobody else at this hackathon will build this — most participants will build email triage, code review, or content moderation (the literal examples from the problem statement)

---

## Why This Idea Wins

### Against the Judging Criteria

**Real-world utility (30% weight):**
On-call incident response is one of the most painful, expensive, high-stakes engineering tasks that exists. Every tech company runs this 24/7. The judges are Meta and HuggingFace engineers — they are personally on-call rotations. This environment solves something they feel viscerally. Score estimate: **28/30**

**Task & grader quality (25% weight):**
Three distinct tasks with ground-truth root causes make graders deterministic. Easy task has one obvious root cause. Medium adds cascading complexity. Hard task (silent data corruption — no alerts) genuinely challenges frontier models. Score estimate: **21/25** (hard task grader needs careful design — see below)

**Environment design (20% weight):**
The living state machine with tick-based degradation is excellent RL design. Role-switching as part of the action space is novel. Partial reward per step is textbook good reward shaping. Score estimate: **19/20**

**Code quality & spec compliance (15% weight):**
Pure Python, easy to containerize, no exotic dependencies. Multi-role observation space adds complexity to the OpenEnv spec definition. Score estimate: **13/15**

**Creativity & novelty (10% weight):**
Full marks. No one has ever built a living infrastructure simulator as an OpenEnv environment. The tick-based degradation + negative rewards for wrong actions is the most interesting reward design judges will see. Score estimate: **10/10**

**Total estimated score: ~94/100**

### Competitive Landscape
Most teams will build:
- Email triage (listed as example in problem statement)
- Code review (listed as example in problem statement)
- Customer support / content moderation (also listed as examples)

These are all valid but will be drowning in lookalike submissions. The judges will have seen 50 email triage environments before they get to ours.

---

## Project Concept — Deep Dive

### The Mental Model
Think of it like a video game where you are the on-call engineer and your city (the production system) is on fire. Every second you wait, more buildings catch fire. You have a radio (the observation) telling you what's happening, and you can call in different specialist teams (role switching) to take actions. If you call the wrong team or give the wrong order, things get worse.

### The "Living" Environment
Most RL environments are static — you get a state, you take an action, you get a new state. Our environment has a `tick()` mechanism. Every step, even before the agent acts, the environment evolves:

```python
def tick(self):
    # Every step, things degrade
    self.state["metrics"]["error_rate"] += 0.02  # errors climb
    
    if self.state["services"]["db"] == "down":
        # cascading: db down → api errors compound
        self.state["metrics"]["error_rate"] += 0.05
        self.state["services"]["api"] = "degraded"
    
    if self.state["services"]["worker"] == "memory_leak":
        # cascading: memory leak → OOM kills → more services down
        self.state["metrics"]["memory"] += 5
        if self.state["metrics"]["memory"] > 95:
            self.state["services"]["worker"] = "down"
```

This creates **real tension**. The agent isn't solving a puzzle at leisure — it's racing against a deteriorating system. This is exactly how real incidents work.

### Wrong Actions Have Consequences
```python
# Example: Agent tries to restart API when DB is actually the problem
if action == "restart service api" and self.state["services"]["db"] == "down":
    # API comes back up momentarily
    self.state["services"]["api"] = "healthy"
    # But 2 steps later it crashes again because root cause (DB) is unfixed
    self.pending_events.append({"step": current_step + 2, "event": "api_crash"})
    reward = -0.1  # penalize wrong diagnosis
    
# Example: Agent scales workers during OOM condition
if action == "scale up worker replicas 3" and self.state["metrics"]["memory"] > 85:
    # More replicas = more memory consumption = worse OOM
    self.state["metrics"]["memory"] += 15
    reward = -0.2  # penalize making things worse
```

---

## The Three Tasks

### Task 1 — Easy: Single Service Outage

**Scenario:**
```
ALERT: API service returning 502 Bad Gateway
ERROR RATE: 34% and climbing
AFFECTED USERS: ~2,400
```

**What the logs show:**
```
[ERROR] 2026-04-01 21:03:12 api-service: Connection refused on port 5432
[ERROR] 2026-04-01 21:03:13 api-service: Database connection pool exhausted
[ERROR] 2026-04-01 21:03:14 api-service: Failed to process request: no DB connection
```

**Root cause:** The PostgreSQL database (port 5432) is down. The API is healthy — it just has no DB to talk to.

**What the agent must do:**
1. Read alerts + logs correctly
2. Identify DB as root cause (not API)
3. Take action: `restart service db`
4. Verify recovery: check that error rate drops

**What a wrong agent does:** Restarts the API (symptom) instead of the database (root cause). API comes back momentarily, crashes again in 2 steps. Score suffers.

**Grader logic:**
```python
def grade_task_1(action_history, final_state):
    score = 0.0
    if "restart service db" in action_history:
        score += 0.5  # correct action taken
    if final_state["services"]["db"] == "healthy":
        score += 0.3  # DB actually recovered
    if final_state["metrics"]["error_rate"] < 0.05:
        score += 0.2  # system actually stable
    return min(score, 1.0)
```

**Difficulty:** Easy. A frontier model should solve this in 3–5 steps if it reads the logs correctly.

---

### Task 2 — Medium: Cascading Failure

**Scenario:**
```
ALERT: 3 services degraded simultaneously
ALERT: CPU at 95%, memory at 89%
ALERT: Multiple 503 errors across API gateway
```

**What the logs show:**
```
[WARN]  2026-04-01 21:15:02 worker-service: Memory usage at 78% (threshold: 80%)
[ERROR] 2026-04-01 21:15:45 worker-service: OOM kill — process terminated
[ERROR] 2026-04-01 21:15:46 auth-service: Dependency worker unavailable
[ERROR] 2026-04-01 21:15:47 api-service: Auth check failed — worker unreachable
[ERROR] 2026-04-01 21:15:48 api-service: Returning 503 to all requests
```

**Root cause:** A memory leak in the `worker-service` caused OOM kills, which cascaded to take down `auth-service` and then `api-service`. 3 services appear down but only 1 is actually broken.

**What the agent must do:**
1. Distinguish cascading failure from simultaneous independent failures
2. Identify `worker-service` as the root cause (the one with the leak)
3. Restart worker (not auth, not api)
4. Verify cascade resolves in the correct order: worker → auth → api

**Correct action sequence:**
```
Step 1: switch role SRE
Step 2: inspect service worker
Step 3: restart service worker
Step 4: verify service auth  ← must wait for worker to come back first
Step 5: verify service api
```

**Why sequence matters:** If agent restarts API first → it comes back for 1 step → crashes again because auth is still down → reward penalized. The grader checks that actions happened in the right order.

**Grader logic:**
```python
def grade_task_2(action_history, final_state, action_timestamps):
    score = 0.0
    
    # Did agent find root cause?
    if "inspect service worker" in action_history:
        score += 0.2
    
    # Did agent fix root cause first?
    worker_restart_step = get_step("restart service worker", action_history)
    api_restart_step = get_step("restart service api", action_history)
    if worker_restart_step and (not api_restart_step or worker_restart_step < api_restart_step):
        score += 0.3  # correct order
    
    # Final state
    if final_state["services"]["worker"] == "healthy":
        score += 0.2
    if final_state["services"]["api"] == "healthy":
        score += 0.2
    if final_state["metrics"]["memory"] < 70:
        score += 0.1
    
    return min(score, 1.0)
```

**Difficulty:** Medium. The cascade pattern trips up models that jump to restart the most visible broken service rather than trace to root cause.

---

### Task 3 — Hard: Silent Data Corruption

**Scenario:**
```
NO CRITICAL ALERTS FIRING
WARN: Error rate at 2.3% (baseline: 0.8%) — within SLA threshold
WARN: P99 latency up 40ms — within SLA threshold  
```

**What the metrics show (agent must correlate these):**
```
error_rate:     [0.8%, 0.9%, 1.1%, 1.4%, 1.8%, 2.3%]  ← steady climb
p99_latency_ms: [120, 125, 134, 148, 167, 193]          ← steady climb
db_query_time:  [12ms, 13ms, 18ms, 26ms, 38ms, 55ms]   ← steady climb
deploy_history: [
    "2026-04-01 20:31:00 — api-service v2.3.1 deployed",  ← 30min ago
    "2026-04-01 19:45:00 — worker-service v1.8.2 deployed"
]
```

**Root cause:** `api-service v2.3.1` introduced a bad database query that creates a slow table scan on every request. No single request fails, but each one is slightly slower + slightly more error-prone. Cumulative degradation with no hard alerts.

**What the agent must do:**
1. Notice the subtle, consistent upward trend across multiple metrics
2. Switch to Dev role to access deploy history
3. Correlate: degradation started ~30 minutes ago, coinciding with `api-service v2.3.1`
4. Trigger rollback: `rollback deploy api-service v2.3.0`
5. Verify metrics stabilize

**Why this is hard:**
- No alerts are firing (agents that only respond to alerts fail immediately)
- Multiple metrics need correlation (agents that look at only one metric miss it)
- Two recent deploys — agent must identify the correct one
- Rollback is an irreversible action — wrong rollback = penalty

**Grader logic:**
```python
def grade_task_3(action_history, final_state, observations_accessed):
    score = 0.0
    
    # Did agent look at metrics trend (not just current snapshot)?
    if "query metrics history" in action_history:
        score += 0.15
    
    # Did agent switch to Dev role and check deploy history?
    if "switch role dev" in action_history and "query deploy history" in action_history:
        score += 0.20
    
    # Did agent rollback the CORRECT service?
    if "rollback deploy api-service" in action_history:
        score += 0.30
        # Penalty if they also unnecessarily rolled back worker
        if "rollback deploy worker-service" in action_history:
            score -= 0.15
    
    # Did metrics stabilize?
    if final_state["metrics"]["error_rate"] < 0.01:
        score += 0.20
    
    # Speed bonus — found it within step budget
    steps_used = len(action_history)
    if steps_used <= 8:
        score += 0.15
    
    return min(max(score, 0.0), 1.0)
```

**Difficulty:** Hard. This genuinely challenges frontier models because it requires proactive investigation, multi-metric correlation, and causal reasoning about deploy timing.

---

## Environment Design

### State Schema
The entire environment state is a Python dictionary. No real infrastructure — pure simulation.

```python
state = {
    "tick": 0,                          # current step number
    "current_role": "sre",              # which role agent is playing
    "incident_id": "INC-2026-0401-042",
    
    "services": {
        "api":     "healthy",           # healthy | degraded | down
        "db":      "down",
        "worker":  "healthy",
        "auth":    "healthy",
        "gateway": "healthy"
    },
    
    "metrics": {
        "error_rate":    0.34,          # 0.0 → 1.0
        "cpu":           0.95,          # 0.0 → 1.0
        "memory":        0.87,          # 0.0 → 1.0
        "p99_latency_ms": 193,
        "requests_per_sec": 1240
    },
    
    "alerts": [                         # list of active alerts
        {"severity": "critical", "message": "API returning 502", "fired_at": "21:03:12"},
        {"severity": "warning",  "message": "DB connection pool exhausted", "fired_at": "21:03:10"}
    ],
    
    "logs": [                           # last N log lines, role-filtered
        "[ERROR] 21:03:14 api-service: Connection refused on port 5432",
        "[ERROR] 21:03:13 api-service: Database connection pool exhausted"
    ],
    
    "deploy_history": [                 # only visible in Dev role
        {"time": "20:31:00", "service": "api-service", "version": "v2.3.1"},
        {"time": "19:45:00", "service": "worker-service", "version": "v1.8.2"}
    ],
    
    "pending_events": [],               # scheduled future state changes
    "episode_done": False,
    "steps_remaining": 15
}
```

### Observation Space
The observation the agent receives is **role-filtered** — different roles see different parts of the state:

```python
def get_observation(self, role: str) -> Observation:
    base = {
        "tick": self.state["tick"],
        "role": role,
        "alerts": self.state["alerts"],
        "services": self.state["services"],
    }
    
    if role == "sre":
        base["metrics"] = self.state["metrics"]
        base["logs"] = self.state["logs"][-10:]      # last 10 lines
        # SRE does NOT see deploy history
        
    elif role == "dev":
        base["logs"] = self.state["logs"][-30:]      # more log history
        base["deploy_history"] = self.state["deploy_history"]
        base["metrics"] = {"error_rate": self.state["metrics"]["error_rate"]}  # limited metrics
        # Dev sees code context, not full infra metrics
        
    elif role == "manager":
        base["sla_status"] = self.compute_sla_status()
        base["affected_users"] = self.estimate_affected_users()
        base["time_since_incident"] = self.state["tick"] * 2  # minutes
        # Manager sees business impact, not raw metrics or logs
    
    return Observation(**base)
```

### Action Space
Actions are strings parsed into structured commands:

```python
VALID_ACTIONS = {
    # Role switching (available in all roles)
    "switch role sre",
    "switch role dev", 
    "switch role manager",
    
    # SRE actions
    "restart service {service_name}",
    "scale up {service_name} replicas {n}",
    "rollback deploy {service_name} {version}",
    "inspect service {service_name}",
    "query metrics history",
    "acknowledge alert {alert_id}",
    
    # Dev actions
    "query deploy history",
    "query logs {service_name} {n_lines}",
    "inspect code diff {service_name}",
    "rollback deploy {service_name} {version}",
    
    # Manager actions
    "escalate incident p1",
    "notify stakeholders {message}",
    "open war room",
    "query sla status",
}
```

### Episode Boundaries
- **Reset:** Randomly selects one of the 3 task scenarios, initializes state, returns initial observation
- **Done conditions:**
  - Agent resolves the incident (all services healthy, error rate < threshold) → success
  - Steps exhausted (max 15 steps per episode) → failure
  - Agent takes a catastrophically wrong action (e.g. `restart service gateway` during a DB incident) → early termination with heavy penalty
- **Max steps:** 15 (designed to run inference in well under 20 minutes)

---

## Reward Function Design

### Per-Step Rewards
```python
REWARDS = {
    # Positive signals
    "correct_diagnosis":          +0.40,   # correctly identified root cause service
    "correct_action_order":       +0.30,   # actions in logical sequence
    "resolution_verified":        +0.20,   # confirmed system actually recovered
    "speed_bonus":                +0.10,   # resolved within 8 steps
    "role_switch_appropriate":    +0.05,   # switched to right role for the action
    
    # Negative signals  
    "wrong_service_restart":      -0.20,   # restarted a healthy service
    "ignored_critical_alert":     -0.10,   # took non-alert action when critical alert firing
    "wrong_rollback":             -0.25,   # rolled back wrong service
    "made_metrics_worse":         -0.15,   # action caused metric to degrade further
    "unnecessary_escalation":     -0.05,   # escalated P1 on an easy task
}
```

### Why This Reward Design Is Good RL
- **Dense signal:** Agent gets reward feedback every step, not just at episode end. This is critical for RL — sparse reward (only +1 at end) makes training extremely slow.
- **Partial credit:** An agent that finds root cause but takes 3 extra steps still scores ~0.7 instead of 0.0. Meaningful gradient signal for learning.
- **Negative rewards for bad behavior:** Prevents the trivially exploitable strategy of "try everything randomly until something works." Shotgun debugging = negative expected return.
- **Speed bonus:** Incentivizes efficient reasoning, not exhaustive search.

### Score Computation (for inference.py)
```python
MAX_TOTAL_REWARD = 1.0  # per task
score = sum(rewards) / MAX_TOTAL_REWARD
score = min(max(score, 0.0), 1.0)  # clamp
```

---

## Multi-Role System

This is the single most novel feature of the environment and the main differentiator from every other submission.

### Why Multiple Roles?
In real incident response, different people have access to different tools and information:
- The SRE can restart services and read metrics but may not have code access
- The developer can read code diffs and deploy history but may not know infrastructure topology
- The manager sees SLA impact and can escalate but doesn't read raw logs

By giving the agent a role-switching mechanism, we force it to reason about *what information it needs* before *what action to take*. This is a significantly harder and more realistic task than "here is all information, now act."

### Role Switching as an Action
Role switching costs one step. This means the agent must plan:
- "Do I need deploy history? → Switch to Dev first"
- "Do I need to restart a service? → I'm already SRE, no switch needed"
- "Do I need to check SLA status before escalating? → Switch to Manager"

Unnecessary role switches waste steps = fewer steps for remediation = lower speed bonus.

### How It Affects Grading
The hard task (silent data corruption) **requires** a role switch to Dev to access deploy history. An agent that stays in SRE role the entire time literally cannot access the information needed to solve it. This is the mechanic that makes the hard task genuinely hard for frontier models.

---

## Technical Architecture

### File Structure
```
devops-warroom/
├── openenv.yaml                    # OpenEnv metadata (auto-generated by openenv init)
├── Dockerfile                      # containerized environment
├── README.md                       # required documentation
├── inference.py                    # required: OpenAI client baseline script (ROOT)
├── requirements.txt
│
├── environment/
│   ├── __init__.py
│   ├── env.py                      # main DevOpsWarRoomEnv class
│   ├── models.py                   # Pydantic typed models (Observation, Action, Reward)
│   ├── state.py                    # state initialization + tick logic
│   ├── actions.py                  # action parser + executor
│   ├── roles.py                    # role switching + observation filtering
│   └── scenarios/
│       ├── __init__.py
│       ├── task_1_single_outage.py
│       ├── task_2_cascading_failure.py
│       └── task_3_silent_corruption.py
│
├── graders/
│   ├── __init__.py
│   ├── grader_task_1.py
│   ├── grader_task_2.py
│   └── grader_task_3.py
│
└── tests/
    ├── test_env.py
    ├── test_graders.py
    └── test_scenarios.py
```

### Core Class: DevOpsWarRoomEnv
```python
from openenv import BaseEnvironment
from .models import Observation, Action, Reward

class DevOpsWarRoomEnv(BaseEnvironment):
    
    def reset(self, task_id: int = None) -> Observation:
        """Initialize environment for a new episode."""
        self.task_id = task_id or random.choice([1, 2, 3])
        self.state = load_scenario(self.task_id)
        self.action_history = []
        self.reward_history = []
        return self.get_observation()
    
    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        """Execute one agent action and advance environment."""
        # 1. Tick the environment (degrade state)
        self.tick()
        
        # 2. Parse and validate action
        parsed = self.parse_action(action.text)
        
        # 3. Execute action → compute reward
        reward = self.execute_action(parsed)
        
        # 4. Check done conditions
        done = self.check_done()
        
        # 5. Update history
        self.action_history.append(action.text)
        self.reward_history.append(reward.value)
        
        # 6. Return new observation
        return self.get_observation(), reward, done, {"task_id": self.task_id}
    
    def state(self) -> dict:
        """Return raw current state (for debugging/inspection)."""
        return self.state.copy()
```

### Pydantic Models (OpenEnv Spec Requirement)
```python
from pydantic import BaseModel
from typing import Optional, List, Dict

class ServiceStatus(BaseModel):
    name: str
    status: str  # "healthy" | "degraded" | "down"

class Alert(BaseModel):
    severity: str   # "critical" | "warning" | "info"
    message: str
    fired_at: str

class Observation(BaseModel):
    tick: int
    current_role: str
    alerts: List[Alert]
    services: Dict[str, str]
    metrics: Optional[Dict[str, float]] = None
    logs: Optional[List[str]] = None
    deploy_history: Optional[List[dict]] = None
    sla_status: Optional[str] = None
    affected_users: Optional[int] = None

class Action(BaseModel):
    text: str   # natural language action string

class Reward(BaseModel):
    value: float    # 0.0 → 1.0 (or negative for penalties)
    reason: str     # human-readable explanation
    done: bool
```

---

## OpenEnv Spec Requirements

These are hard requirements from the problem statement. Missing any one = disqualification.

### 1. openenv.yaml
Auto-generated by `openenv init` but must contain correct metadata:
```yaml
name: devops-warroom
version: 1.0.0
description: "Living infrastructure incident simulator with multi-role agent support"
tags: ["devops", "incident-response", "sre", "real-world"]
tasks:
  - id: task_1
    name: "Single Service Outage"
    difficulty: easy
  - id: task_2
    name: "Cascading Failure"
    difficulty: medium
  - id: task_3
    name: "Silent Data Corruption"
    difficulty: hard
```

### 2. Required API Endpoints
- `POST /reset` → returns initial `Observation`
- `POST /step` → accepts `Action`, returns `(Observation, Reward, done, info)`
- `GET /state` → returns current raw state

### 3. inference.py (CRITICAL — exact format required)
Must be in **root directory**, named exactly `inference.py`.  
Must use **OpenAI client** (not direct Anthropic/HuggingFace SDK).  
Must read from env vars: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`.  
Must emit structured stdout logs in **exact** format:

```python
# Required log format — DO NOT deviate from field names or structure
def log_start(task, env, model):
    print(json.dumps({"type": "START", "task": task, "env": env, "model": model}), flush=True)

def log_step(step, action, reward, done, error):
    print(json.dumps({"type": "STEP", "step": step, "action": action, 
                       "reward": reward, "done": done, "error": error}), flush=True)

def log_end(success, steps, score, rewards):
    print(json.dumps({"type": "END", "success": success, "steps": steps,
                       "score": score, "rewards": rewards}), flush=True)
```

### 4. Dockerfile
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "environment.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 5. Required Environment Variables
```
API_BASE_URL    — LLM API endpoint
MODEL_NAME      — model identifier
HF_TOKEN        — HuggingFace API key
```

### 6. HuggingFace Space Deployment
- Must deploy as a containerized HF Space
- Tagged with `openenv`
- Must respond to `POST /reset` with HTTP 200
- Validated by automated ping during Phase 1

---

## Evaluation Criteria & How We Score

| Parameter | Weight | Our Score | Weighted | Reasoning |
|-----------|--------|-----------|----------|-----------|
| Real-world utility | 30% | 28/30 | 8.4 | On-call SRE pain is real; judges live this |
| Task & grader quality | 25% | 21/25 | 5.25 | Hard task grader needs precise design |
| Environment design | 20% | 19/20 | 3.8 | Living state + role system = excellent design |
| Code quality & spec compliance | 15% | 13/15 | 1.95 | Pure Python, clean; multi-role obs space is tricky |
| Creativity & novelty | 10% | 10/10 | 1.0 | Full marks — nobody builds this |
| **Total** | **100%** | **91/100** | **20.4/22** | **Podium territory** |

### What Gets Us to 97+
The single biggest lever: **nail the hard task grader**. It must be fully deterministic. Define exactly what sequence of observations + actions constitutes "correct multi-metric correlation." That moves task & grader from 21 → 24, pushing total to 97+.

---

## Disqualification Risks & Mitigations

| Risk | How to Avoid |
|------|-------------|
| HF Space doesn't respond to `/reset` | Test deployment 48 hours before deadline |
| `openenv validate` fails | Run it locally before pushing |
| Docker build fails | Test `docker build && docker run` locally |
| inference.py doesn't produce structured logs | Copy log format exactly from sample script |
| Graders always return same score | Test each grader with dummy action sequences |
| Runtime > 20 minutes | Cap episodes at 15 steps; each step is milliseconds in pure Python |
| Doesn't run on 2 vCPU / 8GB RAM | Pure Python sim with no ML models = trivially within limits |
| Plagiarism / copied env | 100% original — novel domain |

---

## File Structure

### What to Generate with openenv init
```bash
openenv init devops-warroom
cd devops-warroom
```
This creates the scaffold. Then fill in:
- `environment/env.py` → main environment class
- `environment/models.py` → Pydantic models
- `graders/` → all three graders
- `inference.py` → in root

### What NOT to Build
- No real Docker containers inside the environment — the entire infrastructure is simulated as Python dicts
- No actual databases, no actual Kubernetes — pure state machine
- No ML models inside the environment — it's the environment the model acts *on*, not a model itself

---

## Implementation Plan

### Day 1–2: Core Environment
- [ ] Run `openenv init devops-warroom`
- [ ] Write `Observation`, `Action`, `Reward` Pydantic models
- [ ] Implement `state.py` — state schema + tick logic for all 3 scenarios
- [ ] Implement `actions.py` — action parser + executor
- [ ] Implement `roles.py` — role switching + observation filtering
- [ ] Basic `step()` / `reset()` / `state()` working

### Day 3: Scenarios + Graders
- [ ] `task_1_single_outage.py` — scenario definition
- [ ] `task_2_cascading_failure.py` — scenario definition
- [ ] `task_3_silent_corruption.py` — scenario definition
- [ ] `grader_task_1.py` — deterministic grader
- [ ] `grader_task_2.py` — deterministic grader with sequence checking
- [ ] `grader_task_3.py` — deterministic grader with role-switch requirement

### Day 4: Reward Function + Testing
- [ ] Implement full reward function with all signals
- [ ] Test: wrong actions produce negative rewards
- [ ] Test: correct action sequence produces score > 0.8
- [ ] Test: graders are deterministic (same input = same score, every time)
- [ ] Test: score variance (easy ~0.8, medium ~0.5, hard ~0.2 for baseline model)

### Day 5: inference.py + Spec Compliance
- [ ] Write `inference.py` using OpenAI client
- [ ] Implement exact `[START]` / `[STEP]` / `[END]` log format
- [ ] Wire `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` from env vars
- [ ] Run `openenv validate` — fix any failures
- [ ] Test: inference script runs end-to-end without error

### Day 6: Docker + HF Deployment
- [ ] Write `Dockerfile`
- [ ] Test: `docker build` succeeds
- [ ] Test: `docker run` starts cleanly
- [ ] Deploy to HuggingFace Space
- [ ] Test: `curl -X POST {space_url}/reset` returns HTTP 200

### Day 7: Polish + Submit
- [ ] Write `README.md` (required: env description, action/obs space, task descriptions, setup, baseline scores)
- [ ] Run pre-submission validation script
- [ ] Run inference script one final time — record baseline scores
- [ ] Submit HF Space URL before **8 April 11:59 PM IST**

---

## Key Constraints

From the problem statement — these are hard limits:

| Constraint | Value | Our Status |
|------------|-------|-----------|
| Inference runtime | < 20 minutes | ✅ Pure Python, ~2–3 min max |
| CPU | 2 vCPU | ✅ No compute-intensive ops |
| Memory | 8 GB RAM | ✅ State is a Python dict |
| Tasks | Minimum 3 | ✅ Exactly 3 (easy/medium/hard) |
| Grader scores | Must be 0.0–1.0 | ✅ Clamped in all graders |
| inference.py location | Root directory | ⚠️ Must not be in a subdirectory |
| LLM calls | Must use OpenAI client | ⚠️ Not direct HF or Anthropic SDK |
| Env vars | API_BASE_URL, MODEL_NAME, HF_TOKEN | ⚠️ Must be wired before submission |
| Log format | Exact [START]/[STEP]/[END] | ⚠️ Any deviation = wrong scoring |
| Real-world task | No games, no toys | ✅ On-call SRE incident response |

---

## Summary

We are building **DevOps War Room** — a living infrastructure incident simulator where an AI agent acts as an on-call SRE with role-switching capabilities. The environment degrades dynamically, wrong actions have real consequences, and the grader rewards correct reasoning over brute-force trial and error.

This idea:
- Is original (no one else will submit this)
- Resonates personally with Meta/HuggingFace judges
- Has deterministic, non-gameable graders
- Has excellent RL properties (dense reward, partial credit, negative penalties)
- Is fully vibe-codeable in pure Python with no exotic dependencies

Estimated score: **94/100**. With a polished hard-task grader: **97/100**. Podium territory.

---

*Last updated: 1 April 2026 | Team BholeChature | Pulkit Pandey + Kanav Kumar*
