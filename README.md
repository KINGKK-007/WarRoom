---
title: DevOps War Room
emoji: 🚨
colorFrom: red
colorTo: gray
sdk: docker
pinned: false
---

# DevOps War Room Simulator

> A living infrastructure incident simulator where an AI agent plays the role of an on-call SRE, diagnosing and resolving production incidents in a dynamically degrading environment — before the system collapses completely.

**Team:** BholeChature — Pulkit Pandey + Kanav Kumar  
**Hackathon:** Meta PyTorch OpenEnv × Scaler School of Technology  
**Framework:** [OpenEnv](https://github.com/huggingface/openenv) (Gymnasium-style API)

---

## What Makes This Unique

1. **Living Environment** — The system degrades every tick via `tick()`. Errors compound, services cascade, and delay is punished. This is not a static puzzle.
2. **Multi-Role Observation** — The agent switches between **SRE**, **Dev**, and **Manager** roles. Each role sees different data and has different action permissions. Correct role selection is part of the challenge.
3. **Wrong Actions Have Consequences** — Restarting the wrong service, rolling back the wrong deploy, or wasting steps all produce negative rewards. Shotgun debugging fails.
4. **Dense Reward Signal** — Partial credit per step, not binary pass/fail. Enables meaningful gradient signal for RL training.

---

## Task Definitions

| Task ID | Name | Difficulty | Root Cause | Key Mechanic |
|---------|------|------------|------------|--------------|
| `task_1` (Easy) | **Single Service Outage** | Easy | PostgreSQL database is DOWN | Direct diagnosis from CRITICAL alert + logs |
| `task_2` (Medium) | **Cascading Failure** | Medium | Worker memory leak → OOM → Auth cascade → API cascade | Sequence-sensitive repair (worker before API) |
| `task_3` (Hard) | **Silent Data Corruption** | Hard | Bad deploy (`api-service v2.3.1`) causing gradual metric degradation | Requires Dev role switch + deploy history correlation |

### Task 1 — Single Service Outage
- **Initial State:** Database DOWN, all other services UP, error_rate 0.0
- **Clues:** CRITICAL alert "Connection refused on port 5432", error logs
- **Solution:** `restart service database`
- **Steps to solve:** 1–3 (SRE role)

### Task 2 — Cascading Failure
- **Initial State:** Worker DEGRADED, all others UP, error_rate 0.05
- **Clues:** WARN log "Memory usage at 78%", no alerts initially
- **Cascade:** Worker DEGRADED → (2 ticks) → Worker DOWN + Auth DEGRADED
- **Solution:** `restart service worker` **before** `restart service api`
- **Steps to solve:** 3–8 (SRE role)

### Task 3 — Silent Data Corruption
- **Initial State:** Database DEGRADED, all others UP, error_rate 0.10
- **Clues:** No alerts, no obvious logs — requires investigation
- **Solution:** `switch role dev` → `rollback deploy api v2.3.1`
- **Steps to solve:** 5–10 (requires role switch to Dev)

---

## Action Space

Actions are natural language strings parsed by a regex-based command parser.

### SRE Role Actions
| Command | Description |
|---------|-------------|
| `restart service {name}` | Restart a broken service (database, api, worker, auth, frontend) |
| `scale {target}` | Scale a service |
| `inspect {target}` | Inspect a service's current state |
| `query metrics` | View current system metrics |

### Dev Role Actions
| Command | Description |
|---------|-------------|
| `rollback deploy {name} {version}` | Rollback a service to a previous version |
| `query deploy` | View deployment history |
| `query logs` | View recent application logs |

### Manager Role Actions
| Command | Description |
|---------|-------------|
| `escalate {target}` | Escalate the incident |
| `notify {target}` | Notify stakeholders |

### Universal Actions
| Command | Description |
|---------|-------------|
| `switch role {sre\|dev\|manager}` | Switch agent's active role (costs 1 tick) |

---

## Observation Space

Observations are role-filtered Pydantic models:

```python
class Observation(BaseModel):
    tick: int                                    # current step number
    current_role: str                            # "SRE" | "Dev" | "Manager"
    services: Dict[str, ServiceState]            # service → "up" | "degraded" | "down"
    alerts: List[Alert]                          # active alerts with severity + message
    metrics: Optional[Dict[str, Any]]            # SRE only: error_rate, etc.
    logs: Optional[List[str]]                    # SRE (last 10) / Dev (last 30)
    deployment_history: Optional[List[str]]      # Dev only
    code_diffs: Optional[List[str]]              # Dev only
    sla_status: Optional[str]                    # Manager only
    estimated_affected_users: Optional[int]      # Manager only
```

---

## Reward Structure

| Signal | Value | Trigger |
|--------|-------|---------|
| Correct diagnosis + restart | +0.40 | Restarting a broken service |
| Wrong service restart | −0.20 | Restarting a healthy service |
| Wrong rollback | −0.25 | Rolling back incorrect target/version |
| Ignored critical alert | −0.10 | Taking a non-fix action during critical alert |
| Unauthorized action | −0.15 | Action not permitted for current role |
| Unrecognizable command | −0.05 | Command that doesn't match any pattern |
| Role switch | 0.00 | Switching roles (costs 1 tick, no reward) |

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/reset` | Reset environment. Body: `{"task_id": "Easy"\|"Medium"\|"Hard"}` |
| `POST` | `/step` | Execute action. Body: `{"action_type": "raw_command", "target": "restart service database"}` |
| `GET` | `/state` | Return current raw environment state |
| `GET` | `/health` | Health check |

---

## Setup & Installation

### Local Development
```bash
pip install -r requirements.txt
python -m pytest tests/ -v           # run test suite
python inference.py                   # run baseline agent (requires LLM API key)
uvicorn environment.server:app --reload  # start API server
```

### Docker
```bash
docker build -t devops-warroom .
docker run -p 8000:8000 devops-warroom
# Test: curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{"task_id": "Easy"}'
```

### Environment Variables
| Variable | Description | Default |
|----------|-------------|---------|
| `API_BASE_URL` | LLM API endpoint URL | `https://api.groq.com/openai/v1` |
| `MODEL_NAME` | Model identifier | `llama-3.1-8b-instant` |
| `HF_TOKEN` | API authentication key (Groq API key) | Required |
| `ENV_URL` | Environment endpoint URL | `https://coolalien35-warroom-deploy.hf.space` |

### HuggingFace Space Deployment

To deploy your own instance with inference capabilities:

1. **Fork/Clone** this repository
2. **Push to HuggingFace Spaces** (Docker SDK):
   ```bash
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/warroom-deploy
   git push space main
   ```

3. **Configure Secrets** in Space settings:
   - Go to: `https://huggingface.co/spaces/YOUR_USERNAME/warroom-deploy/settings`
   - Add these secrets:
     ```
     HF_TOKEN = [your Groq API key from https://console.groq.com]
     API_BASE_URL = https://api.groq.com/openai/v1
     MODEL_NAME = llama-3.1-8b-instant
     ```
   - Click **Factory Reboot** after saving

4. **Verify Deployment**:
   ```bash
   curl https://YOUR_USERNAME-warroom-deploy.hf.space/health
   # Should return: {"status":"healthy"}
   ```

**Get Free Groq API Key**:
1. Visit: https://console.groq.com
2. Sign up (no credit card required)
3. Navigate to API Keys section
4. Create new API key
5. Copy key (format: `gsk_...`)

---

## Baseline Scores

Evaluated on live HuggingFace Space: https://coolalien35-warroom-deploy.hf.space

**Agent**: Groq API (100% free) with `llama-3.1-8b-instant`
**Validator**: OpenEnv-core v0.2.3
**Grading**: Deterministic keyword + state validation (anti-exploit protected)

| Task | Difficulty | Score | Steps | Target | Status |
|------|-----------|-------|-------|--------|--------|
| Task 1 (Single Service Outage) | Easy | **1.00** | 3 | ≥0.65 | ✅ PASS |
| Task 2 (Cascading Failure) | Medium | **1.00** | 3 | ≥0.40 | ✅ PASS |
| Task 3 (Silent Corruption) | Hard | **1.00** | 4 | ≥0.15 | ✅ PASS |

### Example Trajectories

**Task 1 (Database Outage)**:
```
Step 1: restart service database → +0.40 reward (correct diagnosis)
Step 2: query metrics → +0.00 reward (verification)
Step 3: query metrics → +0.00 reward (confirm recovery)
Final: error_rate=0.0, all services healthy
Grader Score: 1.0/1.0 (+0.5 restart +0.5 recovery -0.0 efficiency)
```

**Task 2 (Worker Cascade)**:
```
Step 1: query metrics → +0.00 reward
Step 2: restart service worker → +0.40 reward (root cause identified)
Step 3: query metrics → +0.00 reward (confirm recovery)
Final: error_rate=0.0, cascade prevented
Grader Score: 1.0/1.0 (+0.5 worker +0.5 sequence -0.0 efficiency)
```

**Task 3 (Bad Deployment)**:
```
Step 1: query metrics → +0.00 reward
Step 2: switch role dev → +0.00 reward (role switch)
Step 3: query deploy history → +0.00 reward (investigation)
Step 4: rollback api-service v2.3.0 → -0.40 reward (partial, pending cascade)
Final: Bad deployment rolled back
Grader Score: 1.0/1.0 (+0.3 role switch +0.7 rollback -0.0 efficiency)
```

> **Note**: Rewards during execution are partial/incremental. Final grader scores evaluate full trajectory against task-specific success criteria.

---

## Architecture

```
WarRoom/
├── inference.py              # Baseline LLM agent (OpenAI client)
├── Dockerfile                # Container definition
├── requirements.txt          # Python dependencies
├── openenv.py                # Local BaseEnvironment shim (removed in Docker)
│
├── environment/
│   ├── __init__.py           # Package re-exports
│   ├── env.py                # Core DevOpsWarRoomEnv (step/reset/tick)
│   ├── models.py             # Pydantic contracts (Observation, Action, Reward)
│   ├── actions.py            # Regex command parser
│   ├── roles.py              # Role permission matrix
│   ├── scenarios.py          # Task scenario definitions
│   └── server.py             # FastAPI endpoints
│
├── graders/
│   ├── __init__.py           # Package init
│   ├── task_1.py             # Easy grader — DB restart check
│   ├── task_2.py             # Medium grader — sequence check
│   └── task_3.py             # Hard grader — role switch + rollback check
│
└── tests/
    ├── test_graders.py       # Grader unit tests
    └── test_system.py        # End-to-end system validation
```

---

## License

MIT
