"""
environment/server.py — FastAPI wrapper for DevOpsWarRoomEnv

Exposes the three required OpenEnv endpoints:
  POST /reset   — reset environment to a given task scenario
  POST /step    — execute one agent action
  GET  /state   — return current raw environment state

Plus health-check and root endpoints for HuggingFace Space monitoring.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Any, Dict, Optional
from uuid import uuid4
from threading import Lock
import traceback
import json

from .env import DevOpsWarRoomEnv
from .models import Action, Scenario, ServiceState, Alert

# ---------------------------------------------------------------------------
# App + session registry
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DevOps War Room",
    description="Living infrastructure incident simulator — OpenEnv environment",
    version="1.0.0",
)

# CORS — allow browser-based validators and frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSION_ENVS: Dict[str, DevOpsWarRoomEnv] = {}
SESSION_LOCK = Lock()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: Scenario | str = Scenario.EASY
    seed: Optional[int] = None


class ResetResponse(BaseModel):
    session_id: str
    observation: Dict[str, Any]


class StepRequest(Action):
    session_id: str


class StepResponse(BaseModel):
    session_id: str
    observation: Dict[str, Any]
    reward: Dict[str, Any]
    done: bool
    info: Dict[str, Any]


class StateResponse(BaseModel):
    session_id: str
    state: Dict[str, Any]


def _new_session_id() -> str:
    return f"sess-{uuid4().hex}"


def _create_env_session() -> tuple[str, DevOpsWarRoomEnv]:
    session_id = _new_session_id()
    session_env = DevOpsWarRoomEnv()
    with SESSION_LOCK:
        SESSION_ENVS[session_id] = session_env
    return session_id, session_env


def _get_session_env(session_id: str) -> DevOpsWarRoomEnv:
    with SESSION_LOCK:
        session_env = SESSION_ENVS.get(session_id)
    if session_env is None:
        raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id}")
    return session_env


# ---------------------------------------------------------------------------
# Serialization helper
# ---------------------------------------------------------------------------

def _serialize_state(state: Any) -> Any:
    """Deep-convert nested state objects into JSON-safe Python types."""
    if isinstance(state, ServiceState):
        return state.value
    if isinstance(state, Alert):
        return state.model_dump() if hasattr(state, "model_dump") else state.dict()
    if isinstance(state, dict):
        return {key: _serialize_state(value) for key, value in state.items()}
    if isinstance(state, list):
        return [_serialize_state(item) for item in state]
    return state


def _obs_to_dict(obs) -> dict:
    """Convert an Observation Pydantic model to a plain dict."""
    if hasattr(obs, "model_dump"):
        d = obs.model_dump()
    else:
        d = obs.dict()
    return _serialize_state(d)


def _reward_to_dict(reward) -> dict:
    if hasattr(reward, "model_dump"):
        return _serialize_state(reward.model_dump())
    return _serialize_state(reward.dict())


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    """Root health check — confirms the server is alive."""
    return {
        "status": "ok",
        "environment": "devops-warroom",
        "version": "1.0.0",
    }


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>DevOps War Room Dashboard</title>
  <style>
    :root { --bg:#09111a; --panel:#101b27; --panel-2:#142233; --panel-3:#0d1824; --text:#edf3f8; --muted:#8ea2b5; --good:#33c27f; --warn:#f3b64c; --bad:#ff6b6b; --accent:#62b0ff; --accent-2:#7ef0c4; --line:rgba(255,255,255,.08); }
    * { box-sizing:border-box; }
    body { margin:0; font-family: ui-sans-serif, system-ui, sans-serif; color:var(--text); background:
      radial-gradient(circle at top left, rgba(98,176,255,.16), transparent 28%),
      radial-gradient(circle at top right, rgba(126,240,196,.10), transparent 22%),
      linear-gradient(180deg, #0a121c, #09111a 55%, #071019); }
    .wrap { max-width: 1480px; margin: 0 auto; padding: 24px; }
    .header { display:flex; justify-content:space-between; gap:16px; align-items:flex-start; margin-bottom:18px; }
    .sub { color:var(--muted); margin-top:8px; }
    .controls { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
    .grid { display:grid; grid-template-columns: 1.35fr 1fr 1fr; gap:16px; }
    .panel { background: linear-gradient(180deg, rgba(16,27,39,.96), rgba(12,20,30,.94)); border:1px solid var(--line); border-radius: 18px; padding: 16px; box-shadow: 0 18px 50px rgba(0,0,0,.28); min-width:0; }
    .span-2 { grid-column: span 2; }
    .span-3 { grid-column: span 3; }
    h1,h2,h3 { margin:0 0 12px; }
    h1 { font-size: 32px; }
    .pill { display:inline-flex; align-items:center; padding:6px 10px; border-radius:999px; border:1px solid var(--line); background:rgba(255,255,255,.03); color:var(--muted); font-size:12px; margin-right:8px; margin-bottom:8px; }
    .stats { display:grid; grid-template-columns: repeat(4, 1fr); gap:10px; }
    .metric, .mini, .zone-card, .dep-card { padding: 12px; border-radius: 12px; background:var(--panel-2); border:1px solid var(--line); }
    .metric strong, .mini strong { display:block; color:var(--muted); font-size:12px; margin-bottom:6px; }
    .metric .value { font-size:24px; font-weight:700; }
    .rows { display:flex; flex-direction:column; gap:8px; }
    .row { display:flex; justify-content:space-between; gap:12px; align-items:center; padding:8px 0; border-bottom:1px solid var(--line); }
    .row:last-child { border-bottom:none; }
    .healthy{color:var(--good)} .degraded,.drained{color:var(--warn)} .down,.isolated{color:var(--bad)}
    .bar { height:10px; width:100%; background:#0b1520; border-radius:999px; overflow:hidden; border:1px solid var(--line); }
    .bar > span { display:block; height:100%; background:linear-gradient(90deg, var(--accent), #8dd0ff); }
    .bar.warn > span { background:linear-gradient(90deg, var(--warn), #ffd781); }
    .bar.bad > span { background:linear-gradient(90deg, var(--bad), #ff9b9b); }
    .progress-grid { display:grid; grid-template-columns: repeat(2, 1fr); gap:12px; }
    .progress-title { display:flex; justify-content:space-between; font-size:13px; margin-bottom:6px; color:var(--muted); }
    select, input, button, textarea { border-radius: 12px; border:1px solid var(--line); background:#0c1723; color:var(--text); padding:10px 12px; }
    button { cursor:pointer; background:linear-gradient(180deg, #17304b, #102235); }
    button:hover { filter:brightness(1.05); }
    textarea { width:100%; min-height:84px; resize:vertical; }
    .chart { width:100%; height:154px; background:linear-gradient(180deg, rgba(255,255,255,.02), rgba(255,255,255,.01)); border:1px solid var(--line); border-radius:14px; padding:8px; }
    .timeline, .actions, .alerts { max-height: 340px; overflow:auto; padding-right:6px; }
    .timeline-track { position:relative; padding-left:18px; }
    .timeline-track::before { content:""; position:absolute; left:6px; top:4px; bottom:4px; width:2px; background:linear-gradient(180deg, rgba(98,176,255,.6), rgba(255,255,255,.08)); }
    .event { position:relative; padding:0 0 14px 16px; }
    .event::before { content:""; position:absolute; left:-1px; top:5px; width:10px; height:10px; border-radius:999px; background:var(--accent); box-shadow:0 0 0 4px rgba(98,176,255,.12); }
    .log { font-family: ui-monospace, SFMono-Regular, monospace; font-size:12px; white-space:pre-wrap; color:#c4d2de; }
    .badge { font-size:11px; padding:3px 8px; border-radius:999px; border:1px solid var(--line); color:var(--muted); }
    .action { padding:10px 0; border-bottom:1px solid var(--line); }
    .action:last-child { border-bottom:none; }
    .action-top { display:flex; justify-content:space-between; gap:8px; margin-bottom:4px; }
    .success { color:var(--good); }
    .failure { color:var(--bad); }
    .zones-grid { display:grid; grid-template-columns: repeat(2, 1fr); gap:10px; }
    .zone-card { background:linear-gradient(180deg, rgba(20,34,51,.95), rgba(13,24,36,.95)); }
    .zone-top { display:flex; justify-content:space-between; gap:8px; margin-bottom:10px; }
    .heat { height:10px; border-radius:999px; background:rgba(255,255,255,.05); overflow:hidden; border:1px solid var(--line); }
    .heat span { display:block; height:100%; background:linear-gradient(90deg, var(--good), var(--warn), var(--bad)); }
    .dep-grid { display:grid; grid-template-columns: repeat(3, 1fr); gap:10px; }
    .dep-card { background:linear-gradient(180deg, rgba(20,34,51,.9), rgba(13,24,36,.92)); }
    .dep-name { display:flex; justify-content:space-between; gap:8px; margin-bottom:8px; }
    .dep-links { display:flex; flex-wrap:wrap; gap:6px; }
    .dep-pill { padding:4px 8px; border-radius:999px; border:1px solid var(--line); background:rgba(255,255,255,.03); font-size:11px; color:var(--muted); }
    .sla-ring { display:grid; grid-template-columns: 110px 1fr; gap:14px; align-items:center; }
    .ring { width:110px; height:110px; border-radius:999px; background:conic-gradient(var(--accent) 0deg, var(--accent) 0deg, rgba(255,255,255,.08) 0deg 360deg); display:grid; place-items:center; border:1px solid var(--line); }
    .ring-inner { width:78px; height:78px; border-radius:999px; background:var(--panel-3); display:grid; place-items:center; font-size:20px; font-weight:700; }
    pre { margin:0; }
    details { border-top:1px solid var(--line); padding-top:12px; }
    summary { cursor:pointer; color:var(--muted); }
    @media (max-width: 1150px) {
      .grid, .stats, .progress-grid, .zones-grid, .dep-grid, .sla-ring { grid-template-columns: 1fr; }
      .span-2,.span-3 { grid-column: span 1; }
      .header { flex-direction:column; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="header">
      <div>
        <h1>DevOps War Room</h1>
        <div id="summary" class="sub">Loading state…</div>
      </div>
      <div class="controls">
        <select id="scenario">
          <option value="task_1">Task 1</option>
          <option value="task_2">Task 2</option>
          <option value="task_3">Task 3</option>
          <option value="task_4">Task 4</option>
          <option value="task_5">Task 5</option>
          <option value="task_6">Task 6</option>
          <option value="task_7">Task 7</option>
          <option value="task_8">Task 8</option>
          <option value="task_9">Task 9</option>
          <option value="task_10">Task 10</option>
          <option value="task_11">Task 11</option>
          <option value="Easy">Easy</option>
          <option value="Medium">Medium</option>
          <option value="Hard">Hard</option>
          <option value="EasyRedis">EasyRedis</option>
          <option value="MediumKafka">MediumKafka</option>
          <option value="HardMesh">HardMesh</option>
          <option value="MediumReplica">MediumReplica</option>
          <option value="MediumCache">MediumCache</option>
          <option value="HardRollback">HardRollback</option>
          <option value="HardDNS">HardDNS</option>
          <option value="HardRegion">HardRegion</option>
          <option value="Chaos">Chaos</option>
        </select>
        <input id="seed" type="number" placeholder="Seed (optional)" />
        <button id="resetBtn">Reset Scenario</button>
        <button id="refreshBtn">Refresh</button>
      </div>
    </div>
    <div class="grid">
      <div class="panel span-2">
        <h2>Episode Overview</h2>
        <div id="badges"></div>
        <div class="stats" id="metrics"></div>
      </div>
      <div class="panel">
        <h2>SLA Indicator</h2>
        <div class="sla-ring">
          <div id="slaRing" class="ring"><div id="slaValue" class="ring-inner">0%</div></div>
          <div class="rows">
            <div class="row"><span>Status</span><strong id="slaStatus">n/a</strong></div>
            <div class="row"><span>Target Availability</span><strong id="slaTargetAvailability">n/a</strong></div>
            <div class="row"><span>Target Error Rate</span><strong id="slaTargetError">n/a</strong></div>
            <div class="row"><span>Breaches</span><strong id="slaBreaches">0</strong></div>
          </div>
        </div>
      </div>
      <div class="panel span-2">
        <h2>Reward Graph</h2>
        <div class="mini"><strong>Total Reward</strong><div class="value" id="totalReward">0.0</div></div>
        <div class="mini" style="margin-top:10px;"><strong>Last Reward</strong><div class="value" id="lastReward">n/a</div><div class="sub" id="lastReason"></div></div>
        <svg id="rewardChart" class="chart" viewBox="0 0 520 154" preserveAspectRatio="none"></svg>
      </div>
      <div class="panel">
        <h2>Progress</h2>
        <div class="progress-grid" id="progress"></div>
      </div>
      <div class="panel">
        <h2>Zone Heatmap</h2>
        <div class="zones-grid" id="zones"></div>
      </div>
      <div class="panel">
        <h2>Manual Step</h2>
        <textarea id="command" placeholder="query metrics&#10;restart service postgres-primary"></textarea>
        <div class="controls" style="margin-top:10px;">
          <button id="stepBtn">Send Action</button>
          <span class="badge" id="stepStatus">idle</span>
        </div>
        <div class="sub" id="stepInfo" style="margin-top:10px;"></div>
      </div>
      <div class="panel span-2">
        <h2>Action History</h2>
        <div class="actions" id="actions"></div>
      </div>
      <div class="panel">
        <h2>Active Alerts</h2>
        <div class="alerts" id="alerts"></div>
      </div>
      <div class="panel span-2">
        <h2>Service Dependency Graph</h2>
        <div class="dep-grid" id="dependencyGraph"></div>
      </div>
      <div class="panel">
        <h2>Guidance</h2>
        <div class="rows" id="guidance"></div>
      </div>
      <div class="panel span-2">
        <h2>Timeline Visualization</h2>
        <div class="timeline timeline-track" id="timeline"></div>
      </div>
      <div class="panel">
        <h2>RCA / Last Info</h2>
        <pre id="rca" class="log"></pre>
      </div>
      <div class="panel span-3">
        <details>
          <summary>Raw State Debug View</summary>
          <pre id="raw" class="log" style="margin-top:12px;"></pre>
        </details>
      </div>
    </div>
  </div>
  <script>
    const fmt = (n) => typeof n === 'number' ? (Math.round(n * 100) / 100).toString() : String(n ?? 'n/a');
    const clampPct = (value) => Math.max(0, Math.min(100, value));
    const esc = (value) => String(value ?? '').replace(/[&<>"]/g, (ch) => ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;' }[ch]));
    let currentSessionId = null;

    function rewardPath(history) {
      if (!history.length) {
        return '<text x="24" y="82" fill="#8ea2b5" font-size="12">No rewards yet</text>';
      }
      const width = 520;
      const height = 154;
      const padX = 18;
      const padY = 18;
      const plotWidth = width - padX * 2;
      const plotHeight = height - padY * 2;
      const rewards = history.map((item) => Number(item.reward || 0));
      const maxReward = Math.max(...rewards, 1);
      const minReward = Math.min(...rewards, 0);
      const span = Math.max(maxReward - minReward, 0.15);
      const points = rewards.map((reward, index) => {
        const x = padX + ((plotWidth * index) / Math.max(rewards.length - 1, 1));
        const y = padY + ((maxReward - reward) / span) * plotHeight;
        return `${x},${y}`;
      }).join(' ');
      const last = points.split(' ').pop().split(',');
      const zeroY = padY + ((maxReward - 0) / span) * plotHeight;
      return `
        <line x1="${padX}" y1="${zeroY}" x2="${width - padX}" y2="${zeroY}" stroke="rgba(255,255,255,.12)" stroke-dasharray="4 4"/>
        <polyline fill="none" stroke="#62b0ff" stroke-width="3" points="${points}" />
        <polyline fill="rgba(98,176,255,.12)" stroke="none" points="${padX},${height - padY} ${points} ${width - padX},${height - padY}" />
        <circle cx="${last[0]}" cy="${last[1]}" r="4" fill="#7ef0c4" />
      `;
    }

    function zoneSeverity(detail) {
      const latency = Number(detail.latency_ms || 0);
      const loss = Number(detail.packet_loss || 0);
      if (detail.status === 'down') return 100;
      return clampPct((latency / 4) + (loss * 8) + (detail.drained ? 20 : 0) + (detail.failed_over ? 10 : 0));
    }

    async function resetScenario() {
      const task_id = document.getElementById('scenario').value;
      const seed = document.getElementById('seed').value;
      const response = await fetch('/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task_id, seed: seed ? Number(seed) : null }),
      });
      const data = await response.json();
      currentSessionId = data.session_id;
      await load();
    }

    async function sendStep() {
      const command = document.getElementById('command').value.trim();
      if (!command || !currentSessionId) return;
      document.getElementById('stepStatus').textContent = 'sending';
      const response = await fetch('/step', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: currentSessionId, action_type: 'raw_command', params: { command } }),
      });
      const data = await response.json();
      currentSessionId = data.session_id || currentSessionId;
      document.getElementById('stepStatus').textContent = data.done ? 'done' : 'ok';
      document.getElementById('stepInfo').textContent = data.reward ? `${data.reward.reason} | reward=${data.reward.value}` : JSON.stringify(data);
      await load();
    }

    async function load() {
      if (!currentSessionId) {
        document.getElementById('summary').textContent = 'Reset a scenario to create a session.';
        return;
      }
      const [stateResp, timelineResp] = await Promise.all([
        fetch(`/state?session_id=${encodeURIComponent(currentSessionId)}`),
        fetch(`/timeline?session_id=${encodeURIComponent(currentSessionId)}`),
      ]);
      const statePayload = await stateResp.json();
      const timelinePayload = await timelineResp.json();
      const state = statePayload.state;
      const episode = state.episode || {};
      const incident = state.incident || {};
      const sla = state.sla || {};
      const artifacts = incident.artifact_status || {};
      const rewardHistory = episode.reward_history || [];
      const lastReward = episode.last_reward || {};
      const evidenceDone = (incident.evidence_collected || []).length;
      const evidenceTotal = (incident.required_evidence || []).length || 1;
      const mitigationDone = (incident.mitigations_applied || []).length;
      const mitigationTotal = (incident.required_mitigations || []).length || 1;
      const artifactDone = Object.values(artifacts).filter(Boolean).length;
      const artifactTotal = Object.keys(artifacts).length || 1;
      const stepPct = clampPct(((episode.steps_used || 0) / 24) * 100);
      const slaScore = clampPct(Number(state.metrics?.availability || 0));
      const timelineEvents = (timelinePayload.events || incident.events || []).slice(-18).reverse();
      const targets = new Set([...(incident.service_targets || []), ...Object.keys(state.services || {}).filter((svc) => (incident.blast_radius || []).includes(svc))]);
      const dependencyCards = Object.entries(state.service_details || {})
        .filter(([name]) => targets.size === 0 || targets.has(name))
        .slice(0, 9);

      document.getElementById('summary').textContent = `${incident.name} | severity=${incident.severity} | resolved=${incident.resolved} | affected_users=${state.estimated_affected_users}`;
      document.getElementById('badges').innerHTML = `
        <span class="pill">session ${esc(currentSessionId)}</span>
        <span class="pill">episode ${esc(episode.episode_id ?? 'n/a')}</span>
        <span class="pill">task ${esc(episode.task_id ?? 'n/a')}</span>
        <span class="pill">steps ${esc(episode.steps_used ?? 0)}/24</span>
        <span class="pill">done ${episode.done ? 'yes' : 'no'}</span>
      `;
      document.getElementById('metrics').innerHTML = [
        ['Error Rate', state.metrics.error_rate],
        ['P99 Latency', `${state.metrics.p99_latency_ms} ms`],
        ['Availability', `${state.metrics.availability}%`],
        ['Queue Depth', state.metrics.queue_depth],
        ['CPU', state.metrics.cpu],
        ['Memory', state.metrics.memory],
        ['RPS', state.metrics.requests_per_sec],
        ['Affected Users', state.estimated_affected_users],
      ].map(([k,v]) => `<div class="metric"><strong>${esc(k)}</strong><div class="value">${esc(v)}</div></div>`).join('');

      document.getElementById('totalReward').textContent = fmt(episode.total_reward ?? 0);
      document.getElementById('lastReward').textContent = fmt(lastReward.reward ?? 0);
      document.getElementById('lastReason').textContent = lastReward.reason || 'No reward recorded yet.';
      document.getElementById('rewardChart').innerHTML = rewardPath(rewardHistory.slice(-24));

      document.getElementById('progress').innerHTML = [
        ['Evidence', evidenceDone, evidenceTotal, clampPct((evidenceDone / evidenceTotal) * 100)],
        ['Mitigations', mitigationDone, mitigationTotal, clampPct((mitigationDone / mitigationTotal) * 100)],
        ['Artifacts', artifactDone, artifactTotal, clampPct((artifactDone / artifactTotal) * 100)],
        ['Step Budget', episode.steps_used || 0, 24, stepPct],
      ].map(([label, done, total, pct]) => `
        <div>
          <div class="progress-title"><span>${esc(label)}</span><span>${done}/${total}</span></div>
          <div class="bar ${pct > 80 ? 'warn' : ''}"><span style="width:${pct}%"></span></div>
        </div>
      `).join('');

      document.getElementById('zones').innerHTML = Object.entries(state.zones || {}).map(([zone, detail]) => {
        const severity = zoneSeverity(detail);
        return `
          <div class="zone-card">
            <div class="zone-top">
              <strong>${esc(zone)}</strong>
              <span class="${esc(detail.status)}">${esc(detail.status)}</span>
            </div>
            <div class="progress-title"><span>Heat</span><span>${severity}%</span></div>
            <div class="heat"><span style="width:${severity}%"></span></div>
            <div class="sub">latency=${esc(detail.latency_ms)}ms | loss=${esc(detail.packet_loss)} | drained=${esc(detail.drained)} | failover=${esc(detail.failed_over)}</div>
          </div>
        `;
      }).join('') || '<div class="sub">No zones loaded.</div>';

      document.getElementById('alerts').innerHTML = (state.alerts || []).slice().reverse().map((alert) => `
        <div class="action">
          <div class="action-top"><strong>${esc(alert.source || 'system')}</strong><span class="${alert.severity === 'CRITICAL' ? 'failure' : 'degraded'}">${esc(alert.severity)}</span></div>
          <div>${esc(alert.message)}</div>
        </div>
      `).join('') || '<div class="sub">No active alerts.</div>';

      document.getElementById('guidance').innerHTML = [
        ...(incident.required_evidence || []).filter((item) => !(incident.evidence_collected || []).includes(item)).slice(0,4).map((item) => `<div class="row"><span>Need evidence</span><code>${esc(item)}</code></div>`),
        ...(incident.required_mitigations || []).filter((item) => !(incident.mitigations_applied || []).includes(item)).slice(0,4).map((item) => `<div class="row"><span>Need mitigation</span><code>${esc(item)}</code></div>`),
      ].join('') || '<div class="sub">Required evidence and mitigations are complete.</div>';

      document.getElementById('actions').innerHTML = (incident.actions_executed || []).slice().reverse().map((action) => `
        <div class="action">
          <div class="action-top">
            <strong>${esc(action.action)}${action.target ? ` ${esc(action.target)}` : ''}</strong>
            <span class="${action.success ? 'success' : 'failure'}">${action.success ? 'success' : 'failed'}</span>
          </div>
          <div class="sub">tick=${esc(action.tick)} | params=${esc(JSON.stringify(action.params || {}))}</div>
        </div>
      `).join('') || '<div class="sub">No actions executed yet.</div>';

      document.getElementById('dependencyGraph').innerHTML = dependencyCards.map(([service, detail]) => `
        <div class="dep-card">
          <div class="dep-name">
            <strong>${esc(service)}</strong>
            <span class="${esc(state.services?.[service] || 'healthy')}">${esc(state.services?.[service] || 'unknown')}</span>
          </div>
          <div class="dep-links">
            ${(detail.dependencies || []).slice(0, 8).map((dep) => `<span class="dep-pill">${esc(dep)}</span>`).join('') || '<span class="dep-pill">no deps</span>'}
          </div>
        </div>
      `).join('') || '<div class="sub">No dependency information available.</div>';

      document.getElementById('timeline').innerHTML = timelineEvents.map((evt) => `
        <div class="event">
          <div class="action-top"><strong>${esc(evt.type)}</strong><span class="badge">tick ${esc(evt.tick)}</span></div>
          <div>${esc(evt.summary)}</div>
        </div>
      `).join('') || '<div class="sub">No timeline events yet.</div>';

      document.getElementById('slaStatus').textContent = sla.current_status || 'n/a';
      document.getElementById('slaTargetAvailability').textContent = sla.target_availability != null ? `${sla.target_availability}%` : 'n/a';
      document.getElementById('slaTargetError').textContent = sla.target_error_rate != null ? `${sla.target_error_rate}` : 'n/a';
      document.getElementById('slaBreaches').textContent = String((sla.breaches || []).length);
      document.getElementById('slaValue').textContent = `${slaScore}%`;
      document.getElementById('slaRing').style.background = `conic-gradient(${sla.current_status === 'breached' ? '#ff6b6b' : '#62b0ff'} 0deg, ${sla.current_status === 'breached' ? '#ff6b6b' : '#62b0ff'} ${slaScore * 3.6}deg, rgba(255,255,255,.08) ${slaScore * 3.6}deg 360deg)`;

      document.getElementById('rca').textContent = JSON.stringify(incident.rca || { last_reward: episode.last_reward, last_info: episode.last_info }, null, 2);
      document.getElementById('raw').textContent = JSON.stringify(state, null, 2);
    }
    document.getElementById('resetBtn').addEventListener('click', resetScenario);
    document.getElementById('refreshBtn').addEventListener('click', load);
    document.getElementById('stepBtn').addEventListener('click', sendStep);
    load();
    setInterval(load, 4000);
  </script>
</body>
</html>
"""


@app.get("/health")
def health():
    """Health check endpoint — OpenEnv validator expects status='healthy'."""
    return {"status": "healthy"}


@app.get("/metadata")
def metadata():
    """Return environment metadata (name, description, version)."""
    return {
        "name": "devops-warroom",
        "description": "Living infrastructure incident simulator with multi-role agent support",
        "version": "1.0.0",
        "team": "BholeChature",
        "tags": ["devops", "incident-response", "sre", "real-world"],
    }


@app.get("/schema")
def schema():
    """Return JSON schemas for action, observation, and state."""
    from .models import Observation as ObsModel
    return {
        "reset": ResetRequest.model_json_schema(),
        "step": StepRequest.model_json_schema(),
        "observation": ObsModel.model_json_schema(),
        "state": {
            "type": "object",
            "description": "Raw environment state dict with services, metrics, alerts, logs, etc.",
            "properties": {
                "services": {"type": "object"},
                "metrics": {"type": "object"},
                "alerts": {"type": "array"},
                "logs": {"type": "array"},
                "deploy_history": {"type": "array"},
                "code_diffs": {"type": "array"},
                "sla_status": {"type": "string"},
                "estimated_affected_users": {"type": "integer"},
            },
        },
    }


@app.post("/mcp")
async def mcp_endpoint(request: Request):
    """MCP (Model Context Protocol) JSON-RPC endpoint.

    Supports initialize, tools/list, and tools/call methods.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}, "id": None})

    rpc_id = body.get("id")
    method = body.get("method", "")

    if method == "initialize":
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "devops-warroom", "version": "1.0.0"},
            },
        })

    if method == "tools/list":
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {
                "tools": [
                    {
                        "name": "reset",
                        "description": "Reset the environment to a specific task",
                        "inputSchema": {"type": "object", "properties": {"task_id": {"type": "string", "enum": ["task_1", "task_2", "task_3", "task_4", "task_5", "task_6", "task_7", "task_8", "task_9", "task_10", "task_11", "Easy", "Medium", "Hard", "EasyRedis", "MediumKafka", "HardMesh", "MediumReplica", "MediumCache", "HardRollback", "HardDNS", "HardRegion", "Chaos"]}, "seed": {"type": "integer"}}},
                    },
                    {
                        "name": "step",
                        "description": "Execute an action in the environment",
                        "inputSchema": {"type": "object", "properties": {"session_id": {"type": "string"}, "action_type": {"type": "string"}, "target": {"type": "string"}}},
                    },
                    {
                        "name": "state",
                        "description": "Get the current environment state",
                        "inputSchema": {"type": "object", "properties": {"session_id": {"type": "string"}}},
                    },
                ],
            },
        })

    if method == "tools/call":
        tool_name = body.get("params", {}).get("name", "")
        tool_args = body.get("params", {}).get("arguments", {})

        try:
            if tool_name == "reset":
                task_id = tool_args.get("task_id", "Easy")
                session_id, session_env = _create_env_session()
                obs = session_env.reset(task_id, seed=tool_args.get("seed"), episode_id=session_id)
                result_data = {"session_id": session_id, "observation": _obs_to_dict(obs)}
            elif tool_name == "step":
                action = StepRequest(**tool_args)
                session_env = _get_session_env(action.session_id)
                obs, reward, done, info = session_env.step(action)
                result_data = {
                    "session_id": action.session_id,
                    "observation": _obs_to_dict(obs),
                    "reward": _reward_to_dict(reward),
                    "done": done,
                    "info": _serialize_state(info),
                }
            elif tool_name == "state":
                session_env = _get_session_env(str(tool_args.get("session_id", "")))
                result_data = {"session_id": tool_args.get("session_id"), "state": _serialize_state(session_env.state)}
            else:
                return JSONResponse({"jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}})

            return JSONResponse({"jsonrpc": "2.0", "id": rpc_id, "result": [{"type": "text", "text": json.dumps(result_data)}]})
        except Exception as e:
            return JSONResponse({"jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32000, "message": str(e)}})

    return JSONResponse({"jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32601, "message": f"Method not found: {method}"}})


@app.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest | None = None):
    """Reset the environment against a specific task scenario.

    Returns the initial observation as a JSON dict.
    """
    if request is None:
        request = ResetRequest()
    try:
        session_id, session_env = _create_env_session()
        obs = session_env.reset(request.task_id, seed=request.seed, episode_id=session_id)
        return {"session_id": session_id, "observation": _obs_to_dict(obs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@app.post("/step", response_model=StepResponse)
def step(action: StepRequest):
    """Execute one agent action in the environment.

    Accepts an Action model and returns:
      - observation: the new environment observation
      - reward: {value, reason, done}
      - done: whether the episode has ended
      - info: additional metadata
    """
    try:
        session_env = _get_session_env(action.session_id)
        obs, reward, done, info = session_env.step(action)
        return {
            "session_id": action.session_id,
            "observation": _obs_to_dict(obs),
            "reward": _reward_to_dict(reward),
            "done": done,
            "info": _serialize_state(info),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Step failed: {str(e)}\n{traceback.format_exc()}",
        )


@app.get("/state", response_model=StateResponse)
def get_state(session_id: str):
    """Return the current raw environment state for debugging/inspection.

    All Pydantic models and enums are serialized to plain JSON types.
    """
    try:
        session_env = _get_session_env(session_id)
        return {"session_id": session_id, "state": _serialize_state(session_env.state)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"State retrieval failed: {str(e)}")


@app.get("/timeline")
def get_timeline(session_id: str):
    try:
        session_env = _get_session_env(session_id)
        return {"session_id": session_id, "events": session_env.timeline()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Timeline retrieval failed: {str(e)}")
