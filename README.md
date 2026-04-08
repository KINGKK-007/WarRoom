---
title: DevOps War Room
emoji: "🚨"
colorFrom: red
colorTo: gray
sdk: docker
pinned: false
---

# DevOps War Room

DevOps War Room is an OpenEnv production incident-response simulator. Agents operate as on-call responders across `SRE`, `Dev`, and `Manager` roles, investigate live failures, apply mitigations, verify SLA recovery, communicate status, and produce RCA and postmortem artifacts.

**Team:** BholeChature  
**Repository:** https://github.com/KINGKK-007/WarRoom  
**Hugging Face Space:** https://coolalien35-warroom-deploy.hf.space  
**OpenEnv Spec File:** [openenv.yaml](./openenv.yaml)

## Quickstart

### Local server

```bash
pip install -r requirements.txt
uvicorn environment.server:app --host 0.0.0.0 --port 8000
```

### Deterministic baseline

```bash
export ENV_URL="http://localhost:8000"
export BASELINE_SEED=7
python inference.py
```

### Adaptive baseline

```bash
export ENV_URL="http://localhost:8000"
export BASELINE_SEED=7
python adaptive_inference.py --task task_3
```

### Tests

```bash
pytest -q
```

Current local result:

- `48 passed`

## Environment Summary

This repository models a living production environment rather than a single static puzzle:

- 40 named services across data, app, edge, infra, observability, and ops layers
- 4 availability zones with `drain`, `failover`, and `restore` mechanics
- dependency propagation and blast radius expansion
- delayed recoveries, failed mitigations, and partial mitigation effects
- role-filtered observations for SRE, Dev, and Manager views
- seeded variants and a procedural `Chaos` scenario
- deterministic graders and bounded dense rewards
- a lightweight browser dashboard for inspection and debugging

## Architecture

Text architecture diagram:

1. `inference.py` or `adaptive_inference.py` calls the FastAPI server in [environment/server.py](./environment/server.py).
2. `/reset` creates a fresh session-scoped [DevOpsWarRoomEnv](./environment/env.py) instance and returns `session_id` plus the first observation.
3. `/step` routes the action to that session environment, applies the mutation, advances `tick()`, computes reward, and returns `observation`, `reward`, `done`, and `info`.
4. The environment uses:
   - [environment/scenarios.py](./environment/scenarios.py) for scenario templates and seeded variants
   - [environment/actions.py](./environment/actions.py) for command parsing
   - [environment/roles.py](./environment/roles.py) for role permissions
   - [environment/models.py](./environment/models.py) for typed Observation / Action / Reward models
5. Deterministic graders in [graders/](./graders) score final state quality.
6. The dashboard in `/dashboard` visualizes reward history, SLA, zones, dependencies, timeline, and action history for an active session.

## Scenario Progression

The project now has two layers of progression:

- Core benchmark progression: `task_1` through `task_6`
- Advanced extension set: `task_7` through `task_11`

The core progression is:

| Task ID | Scenario | Difficulty | Description |
|---|---|---|---|
| `task_1` | `Easy` | easy | Primary database outage cascading into auth and billing failures |
| `task_2` | `Medium` | medium | Worker backlog plus zone degradation |
| `task_3` | `Hard` | hard | Bad api-gateway deploy combined with zone failure |
| `task_4` | `EasyRedis` | easy | Session-store outage affecting login and cart restore |
| `task_5` | `MediumKafka` | medium | Kafka broker partition plus consumer lag |
| `task_6` | `HardMesh` | hard | Service-mesh certificate regression plus zone outage |

Advanced graded scenarios:

| Task ID | Scenario | Difficulty | Description |
|---|---|---|---|
| `task_7` | `MediumReplica` | medium | Database replication lag degrading read-heavy services |
| `task_8` | `MediumCache` | medium | Cache stampede on hot keys and overloaded cart path |
| `task_9` | `HardRollback` | hard | Partial deploy rollback failure leaving stale config live |
| `task_10` | `HardDNS` | hard | DNS control-plane outage across the edge request path |
| `task_11` | `HardRegion` | hard | Cascading region failure across multiple east-region zones |

Bonus procedural scenario:

| Scenario | Difficulty | Description |
|---|---|---|
| `Chaos` | advanced | Procedural overlapping service, deploy, and network failures |

## Scenario Catalog

Canonical named scenarios implemented in [environment/scenarios.py](./environment/scenarios.py):

- `Easy`: `postgres-primary` down, downstream auth and billing degradation
- `Medium`: worker backlog, async service degradation, zone impairment
- `Hard`: bad `api-gateway` deploy plus zone outage and retry amplification
- `EasyRedis`: `redis-session` failure affecting login and session-backed flows
- `MediumKafka`: kafka broker partition with worker lag and async blast radius
- `HardMesh`: `service-mesh` cert regression breaking edge mTLS flows
- `MediumReplica`: replica lag causing stale reads and read-path latency
- `MediumCache`: cache stampede driving hot-key misses and queue pressure
- `HardRollback`: rollback only partially lands until a clean restart occurs
- `HardDNS`: DNS control-plane outage plus degraded edge zone
- `HardRegion`: multi-zone regional failure with coordinated recovery demands
- `Chaos`: seeded procedural multi-root-cause incident generation

Examples:

```bash
curl -s -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id":"task_4","seed":7}'
```

```bash
curl -s -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id":"HardMesh"}'
```

```bash
curl -s -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id":"task_10","seed":11}'
```

## Observation Space

Observations are typed Pydantic models from [environment/models.py](./environment/models.py).

Shared observation fields:

- `tick`
- `current_role`
- `services`
- `alerts`
- `zone_health`
- `service_distribution`
- `incident_summary`
- `available_actions`
- `suggested_actions`
- `steps_remaining`

Role-specific visibility:

- `SRE`: metrics, logs, traces, metrics history
- `Dev`: deployment history, code diffs, logs, traces
- `Manager`: SLA status, affected-user estimate, incident-oriented logs

`incident_summary` includes current task id, episode id, service targets, zone targets, completed evidence, remaining mitigations, artifact status, SLA status, affected users, and seeded variant metadata.

## Action Space

The action surface is operational rather than synthetic.

Diagnosis:

- `inspect`
- `query metrics`
- `query logs`
- `query traces`
- `query topology`
- `query deploy history`
- `run health check`

Mitigations:

- `restart service`
- `rollback`
- `scale`
- `clear queue`
- `tune autoscaling`
- `failover zone`
- `drain zone`
- `restore zone`
- `rebalance traffic`
- `throttle service`
- `isolate service`

Incident workflow:

- `acknowledge alert`
- `verify sla`
- `run rca`
- `generate postmortem`
- `attach runbook`
- `notify stakeholders`
- `update status page`
- `escalate leadership`
- `switch role {SRE|Dev|Manager}`

## Session-Based API

The server is session-scoped. Each `POST /reset` creates a unique environment instance and returns a `session_id`. All later `step`, `state`, and `timeline` calls must use that session id.

Endpoints:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/reset` | create a new session and return the initial observation |
| `POST` | `/step` | execute one action against a session |
| `GET` | `/state?session_id=...` | fetch raw state for a session |
| `GET` | `/timeline?session_id=...` | fetch structured incident events for a session |
| `GET` | `/health` | health check |
| `GET` | `/metadata` | environment metadata |
| `GET` | `/schema` | request and observation schemas |
| `GET` | `/dashboard` | interactive browser dashboard |

### Reset example

```bash
curl -s -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id":"task_3","seed":7}'
```

Example response shape:

```json
{
  "session_id": "sess-...",
  "observation": {
    "tick": 0,
    "current_role": "SRE"
  }
}
```

### Step example

```bash
curl -s -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"session_id":"sess-...","action_type":"raw_command","params":{"command":"query metrics"}}'
```

### State example

```bash
curl -s "http://localhost:8000/state?session_id=sess-..."
```

## Reward Design

Per-step rewards are bounded to `[0.0, 1.0]`.

The reward function in [environment/env.py](./environment/env.py) combines:

- partial progress on `error_rate`, latency, availability, and queue depth
- progress on recovering required services and zones
- progress on artifact completion
- bonuses for required evidence and required mitigations
- bonuses for production follow-through such as `verify_sla`, `run_rca`, `attach_runbook`, and `generate_postmortem`
- penalties for unknown actions, unauthorized actions, repeated actions, unnecessary restarts, ignored critical alerts, and irrelevant mitigations

This means the reward is dense enough to shape behavior, but the final benchmark still depends on deterministic grader scores.

## Graders

Graders are deterministic and return scores in `[0.0, 1.0]`.

The grading stack is:

- shared grading logic in [graders/common.py](./graders/common.py)
- per-task wrappers in [graders/task_1.py](./graders/task_1.py) through [graders/task_11.py](./graders)
- procedural grading in [graders/chaos.py](./graders/chaos.py)

The graders score:

- target service recovery
- target zone recovery
- final metrics
- required evidence collected
- required mitigations applied
- artifact completion
- communication actions when required
- SLA verification

They also apply deterministic penalties for ignored alerts, repeated wasteful actions, wrong restarts, and wrong rollbacks.

## Inference / Baselines

[inference.py](./inference.py) and [adaptive_inference.py](./adaptive_inference.py) are both observation-driven. They do not use a scripted golden-path playbook.

Shared policy behavior:

- reads live observations and raw state
- ranks candidate actions from evidence gaps, mitigation gaps, role constraints, degraded services, zone status, and missing artifacts
- uses deterministic seeds for reproducible runs
- logs strict `START`, `STEP`, and `END` records

Usage:

```bash
export ENV_URL="http://localhost:8000"
export BASELINE_SEED=7
python inference.py --task task_5
```

```bash
export ENV_URL="http://localhost:8000"
export BASELINE_SEED=7
python adaptive_inference.py --task task_10
```

## Environment Variables

| Variable | Purpose | Default |
|---|---|---|
| `ENV_URL` | environment base URL | `https://coolalien35-warroom-deploy.hf.space` |
| `BASELINE_SEED` | deterministic environment seed used by the baselines | `7` |
| `REQUEST_TIMEOUT_S` | request timeout for API calls | `30` |
| `MAX_EPISODE_STEPS` | max baseline steps | `24` |
| `MODEL_NAME` | label used in structured baseline logs | `deterministic-baseline` or `adaptive-baseline` |
| `API_BASE_URL` | optional OpenAI-compatible endpoint reserved for fallback mode | `https://api.groq.com/openai/v1` |
| `HF_TOKEN` | optional API key reserved for fallback mode | empty |
| `ALLOW_LLM_FALLBACK` | enable optional model fallback if supported | `0` |

Notes:

- The default baseline path is deterministic and does not require `HF_TOKEN`.
- `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` are retained for compatibility with optional future fallback usage.

## Dashboard

The dashboard is served from `/dashboard` and is intentionally lightweight. It currently includes:

- scenario selector with seeded reset
- reward graph
- SLA indicator
- progress bars for evidence, mitigations, artifacts, and step budget
- zone heatmap
- service dependency graph
- timeline visualization
- action history viewer
- active alerts
- manual action stepping
- raw state inspector

It is useful for debugging baseline behavior and reviewing incident trajectories during development.

## Evaluation

Evaluation happens in two layers:

1. Dense per-step reward from the environment
2. Deterministic final-state score from the appropriate grader

Recommended evaluation workflow:

```bash
pytest -q
python inference.py --task task_1
python adaptive_inference.py --task task_6
```

Representative current baseline results from the maintained code path:

- `task_1`: success, score around `0.94`
- `task_2`: success, score around `0.84`
- `task_4`: success, score around `0.94`
- `task_5`: success, score around `0.92`
- `task_6`: success, score around `0.94`
- `task_3`: deterministic partial recovery around `0.79` on the referenced seed

## Reproducibility

Reproducibility is built into the runtime:

- `/reset` accepts an optional `seed`
- sessions are isolated per client
- same task id plus same seed yields the same variant family
- reward values are deterministic for a fixed action sequence
- graders are deterministic pure scoring functions

Recommended reproducibility flow:

```bash
export ENV_URL="http://localhost:8000"
export BASELINE_SEED=123
python inference.py --task task_2
python inference.py --task task_2
```

Two runs with the same seed should follow the same deterministic environment variant.

## Docker

Build locally:

```bash
docker build -t devops-warroom .
```

Run locally:

```bash
docker run -p 8000:7860 devops-warroom
```

Smoke test:

```bash
curl -s http://localhost:8000/health
```

## Hugging Face Deployment

The Space is configured for Docker deployment through the front matter in this README and the project [Dockerfile](./Dockerfile).

Typical deployment flow:

```bash
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/warroom-deploy
git push space main
```

Health check:

```bash
curl -s https://YOUR_USERNAME-warroom-deploy.hf.space/health
```

## OpenEnv Notes

The implementation is structured for OpenEnv-style use:

- typed `Observation`, `Action`, and `Reward` models
- public `reset`, `step`, and `state` endpoints
- deterministic graders
- bounded per-step rewards
- task metadata in [openenv.yaml](./openenv.yaml)

The actual `openenv validate` CLI was not run in this workspace, so validator success is not claimed here.

## Repository Layout

```text
WarRoom/
├── environment/
│   ├── actions.py
│   ├── env.py
│   ├── models.py
│   ├── roles.py
│   ├── scenarios.py
│   └── server.py
├── graders/
│   ├── common.py
│   ├── task_1.py
│   ├── task_2.py
│   ├── task_3.py
│   ├── task_4.py
│   ├── task_5.py
│   ├── task_6.py
│   ├── task_7.py
│   ├── task_8.py
│   ├── task_9.py
│   ├── task_10.py
│   ├── task_11.py
│   └── chaos.py
├── tests/
│   ├── conftest.py
│   ├── test_graders.py
│   └── test_system.py
├── baseline_policy.py
├── inference.py
├── adaptive_inference.py
├── benchmark.py
├── openenv.yaml
├── Dockerfile
└── README.md
```
