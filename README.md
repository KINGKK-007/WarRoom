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
| `API_BASE_URL` | LLM API endpoint URL | `https://api.openai.com/v1` |
| `MODEL_NAME` | Model identifier | `gpt-4-turbo` |
| `HF_TOKEN` | HuggingFace / API authentication key | `dummy-key` |

---

## Baseline Scores (GPT-4 Turbo)

| Task | Score | Steps | Notes |
|------|-------|-------|-------|
| Task 1 (Easy) | **~0.90** | 1–3 | Direct restart from alert reading |
| Task 2 (Medium) | **~0.60** | 4–8 | Requires correct restart sequence |
| Task 3 (Hard) | **~0.20** | 6–12 | Requires role switch + version correlation |

**Aggregate baseline:** ~0.57 (average across 3 tasks)

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
