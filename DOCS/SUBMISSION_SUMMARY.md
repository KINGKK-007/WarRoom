# Submission Summary

## Baseline Results

| Task | Score | Playbook Summary |
|---|---:|---|
| Task 1 | `0.97` | Acknowledge alert, inspect `postgres-primary`, verify with metrics/logs/health check, restart, verify SLA, generate RCA and postmortem. |
| Task 2 | `1.00` | Diagnose `worker-service`, inspect observability surfaces, clear the queue, scale and tune autoscaling, heal `us-east-1b`, verify SLA, generate RCA and postmortem. |
| Task 3 | `0.94` | Diagnose the `api-gateway` deploy regression, inspect metrics/traces/topology/deploys, rollback to `v3.2.0`, fail over and restore `us-east-1c`, verify SLA, generate RCA and postmortem. |

## Adaptive Baseline

`adaptive_inference.py` infers the incident type from live observation and follows a role-aware recovery flow. On the three official tasks it currently matches the deterministic baseline:

| Task | Score |
|---|---:|
| Task 1 | `0.97` |
| Task 2 | `1.00` |
| Task 3 | `0.94` |

## Reproducibility

### Local server

```bash
pip install -r requirements.txt
uvicorn environment.server:app --host 0.0.0.0 --port 8000
```

### Baseline agent against local server

```bash
export HF_TOKEN="your_token"
export ENV_URL="http://localhost:8000"
python inference.py
```

### Baseline agent against Hugging Face Space

```bash
export HF_TOKEN="your_token"
unset ENV_URL
python inference.py
```

`inference.py` defaults to the deployed Space:

```python
ENV_URL = os.environ.get("ENV_URL", "https://coolalien35-warroom-deploy.hf.space")
```

Variant and dashboard checks:

```bash
curl -s -X POST https://coolalien35-warroom-deploy.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id":"Hard","seed":7}'
```

```bash
curl -s https://coolalien35-warroom-deploy.hf.space/timeline
open https://coolalien35-warroom-deploy.hf.space/dashboard
```

## Secret Chaos Task

The environment also exposes a procedural `Chaos` scenario beyond the three official graded tasks.

Properties:

- seeds at least 5 overlapping root causes
- combines service outages, zone failures, and bad deploys
- exercises the dependency graph and multi-zone cascade logic
- forces mixed mitigation across restart, rollback, queue, and zone recovery workflows

Examples of concurrent failure types:

- database outage
- worker queue explosion
- zone network partition
- bad `api-gateway` deploy
- service-mesh degradation

## SLA Breach Termination

Episodes terminate early when the incident becomes operationally unacceptable.

Current terminal conditions include:

- `3+` recorded SLA breaches
- availability dropping below the configured floor
- unrecoverable error-rate explosion
- normal full recovery completion

This prevents agents from farming dense reward indefinitely after a failed response.

## Ignored Alert Penalties

The reward model penalizes agents that do not react to critical alerts.

Mechanic:

- critical alerts increment `ignored_alert_ticks` when the agent avoids acknowledgement or diagnostic response
- graders subtract score for alert neglect
- dense reward also becomes less favorable as alert neglect grows

This blocks passive looping strategies such as repeated metrics polling during active incidents.
