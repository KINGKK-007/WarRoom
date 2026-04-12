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
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>DevOps War Room — 3D Service Graph</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:#0a0a0c;font-family:ui-sans-serif,system-ui,sans-serif;color:#edf3f8;overflow:hidden}
  #graph{width:100vw;height:100vh}
  /* ── Glass panel shared ── */
  .glass{
    position:absolute;background:rgba(10,10,12,.55);
    border:1px solid rgba(255,255,255,.1);border-radius:16px;
    padding:16px;backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);
    box-shadow:0 8px 32px rgba(0,0,0,.6),inset 0 1px 0 rgba(255,255,255,.06);
    z-index:10;
  }
  /* ── Left panel ── */
  #info{top:20px;left:20px;width:240px}
  #info h2{font-size:15px;font-weight:700;margin-bottom:10px;letter-spacing:.04em}
  #status{font-size:11px;color:#8ea2b5;margin-bottom:12px;line-height:1.5}
  .legend-row{display:flex;align-items:center;gap:8px;font-size:12px;color:#8ea2b5;margin:5px 0}
  .dot{width:10px;height:10px;border-radius:50%;flex-shrink:0}
  #node-detail{margin-top:12px;padding-top:12px;border-top:1px solid rgba(255,255,255,.08);display:none}
  #node-detail h3{font-size:13px;font-weight:700;margin-bottom:6px}
  .nd-row{font-size:11px;color:#8ea2b5;margin:3px 0}
  /* ── Right controls ── */
  #controls{top:20px;right:20px;width:210px;display:flex;flex-direction:column;gap:8px}
  select,input{width:100%;padding:8px 10px;border-radius:10px;border:1px solid rgba(255,255,255,.1);
    background:rgba(255,255,255,.05);color:#edf3f8;font-size:12px;outline:none}
  button{width:100%;padding:8px 10px;border-radius:10px;border:1px solid rgba(255,255,255,.1);
    background:rgba(255,255,255,.07);color:#edf3f8;font-size:12px;cursor:pointer;transition:background .15s}
  button:hover{background:rgba(255,255,255,.13)}
  /* ── Bottom metrics bar ── */
  #metrics{bottom:20px;left:50%;transform:translateX(-50%);
    display:flex;gap:20px;padding:12px 24px;white-space:nowrap}
  .metric{text-align:center}
  .metric .val{font-size:18px;font-weight:700}
  .metric .lbl{font-size:10px;color:#8ea2b5;margin-top:2px}
</style>
</head>
<body>

<!-- Left info panel -->
<div class="glass" id="info">
  <h2>⚡ War Room — 3D Graph</h2>
  <div id="status">Reset a scenario to begin.</div>
  <div class="legend-row"><div class="dot" style="background:#00ff88;box-shadow:0 0 6px #00ff88"></div>Healthy</div>
  <div class="legend-row"><div class="dot" style="background:#ffaa00;box-shadow:0 0 6px #ffaa00"></div>Degraded</div>
  <div class="legend-row"><div class="dot" style="background:#ff0044;box-shadow:0 0 6px #ff0044"></div>Down</div>
  <div class="legend-row"><div class="dot" style="background:#62b0ff;box-shadow:0 0 6px #62b0ff"></div>Restarting</div>
  <div class="legend-row"><div class="dot" style="background:#cc44ff;box-shadow:0 0 6px #cc44ff"></div>Isolated</div>
  <div id="node-detail">
    <h3 id="nd-name"></h3>
    <div class="nd-row">State: <b id="nd-state"></b></div>
    <div class="nd-row">Tier: <span id="nd-tier"></span></div>
    <div class="nd-row" id="nd-cpu-row">CPU: <span id="nd-cpu"></span></div>
    <div class="nd-row" id="nd-mem-row">Mem: <span id="nd-mem"></span></div>
  </div>
</div>

<!-- Right controls -->
<div class="glass" id="controls">
  <select id="scenario">
    <option value="task_1">Task 1 — Postgres Outage</option>
    <option value="task_2">Task 2 — Queue / Zone</option>
    <option value="task_3">Task 3 — Bad Deploy</option>
    <option value="task_4">Task 4 — Redis Session</option>
    <option value="task_5">Task 5 — Kafka</option>
    <option value="task_6">Task 6 — Mesh Cert</option>
    <option value="task_7">Task 7 — Replica Lag</option>
    <option value="task_8">Task 8 — Cache Stampede</option>
    <option value="task_9">Task 9 — Rollback Fail</option>
    <option value="task_10">Task 10 — DNS</option>
    <option value="task_11">Task 11 — Region Fail</option>
    <option value="Chaos">Chaos</option>
  </select>
  <input id="seed" type="number" placeholder="Seed (optional)"/>
  <button id="resetBtn">Reset Scenario</button>
  <button id="refreshBtn">Refresh State</button>
  <button id="rotateBtn">Stop Rotation</button>
</div>

<!-- Bottom metrics -->
<div class="glass" id="metrics">
  <div class="metric"><div class="val" id="m-healthy">—</div><div class="lbl">Healthy</div></div>
  <div class="metric"><div class="val" id="m-degraded">—</div><div class="lbl">Degraded</div></div>
  <div class="metric"><div class="val" id="m-down">—</div><div class="lbl">Down</div></div>
  <div class="metric"><div class="val" id="m-rps">—</div><div class="lbl">RPS</div></div>
  <div class="metric"><div class="val" id="m-err">—</div><div class="lbl">Error Rate</div></div>
  <div class="metric"><div class="val" id="m-avail">—</div><div class="lbl">Availability</div></div>
</div>

<div id="graph"></div>

<!-- CDN deps -->
<script src="https://unpkg.com/three@0.160.0/build/three.min.js"></script>
<script src="https://unpkg.com/three@0.160.0/examples/js/postprocessing/EffectComposer.js"></script>
<script src="https://unpkg.com/three@0.160.0/examples/js/postprocessing/RenderPass.js"></script>
<script src="https://unpkg.com/three@0.160.0/examples/js/postprocessing/ShaderPass.js"></script>
<script src="https://unpkg.com/three@0.160.0/examples/js/postprocessing/UnrealBloomPass.js"></script>
<script src="https://unpkg.com/three@0.160.0/examples/js/shaders/LuminosityHighPassShader.js"></script>
<script src="https://unpkg.com/three@0.160.0/examples/js/shaders/CopyShader.js"></script>
<script src="https://unpkg.com/3d-force-graph@1.73.0/dist/3d-force-graph.min.js"></script>
<script src="https://unpkg.com/gsap@3.12.5/dist/gsap.min.js"></script>

<script>
// ── Service dependency map ────────────────────────────────────────────────────
const DEPS = {
  "postgres-primary":[],
  "postgres-replica":["postgres-primary"],
  "redis-cache":[],
  "redis-session":[],
  "kafka":[],
  "zookeeper":[],
  "mongodb":[],
  "clickhouse":[],
  "elasticsearch":[],
  "object-storage":[],
  "config-service":[],
  "dns-control-plane":[],
  "service-mesh":[],
  "api-gateway":["edge-proxy","service-mesh","auth-service","dns-control-plane"],
  "edge-proxy":["service-mesh","dns-control-plane"],
  "auth-service":["redis-session","postgres-primary","dns-control-plane"],
  "user-service":["postgres-primary","redis-cache"],
  "profile-service":["user-service","mongodb"],
  "billing-service":["postgres-primary","kafka"],
  "order-service":["postgres-primary","inventory-service","payment-service"],
  "inventory-service":["postgres-primary","redis-cache"],
  "payment-service":["billing-service","fraud-service"],
  "cart-service":["redis-cache","user-service"],
  "recommendation-service":["clickhouse","kafka"],
  "search-service":["elasticsearch"],
  "notification-service":["kafka","email-service"],
  "email-service":["object-storage"],
  "analytics-service":["kafka","clickhouse"],
  "report-service":["analytics-service","clickhouse"],
  "worker-service":["kafka","redis-cache"],
  "scheduler-service":["worker-service","config-service"],
  "fraud-service":["postgres-primary","mongodb"],
  "frontend-web":["api-gateway","search-service"],
  "mobile-bff":["api-gateway","user-service"],
  "admin-portal":["api-gateway","report-service"],
  "prometheus":[],
  "grafana":["prometheus"],
  "loki":["object-storage"],
  "tempo":["object-storage"],
  "status-page":[]
};

const STATE_COLOR = {
  healthy:"#00ff88", degraded:"#ffaa00", down:"#ff0044",
  restarting:"#62b0ff", isolated:"#cc44ff"
};

function getTier(n){
  if(/postgres|redis|kafka|mongodb|clickhouse|elasticsearch/.test(n)) return "data";
  if(/gateway|proxy|frontend|mobile|admin/.test(n))                   return "edge";
  if(/prometheus|grafana|loki|tempo/.test(n))                         return "observability";
  if(/config|dns|mesh|zookeeper|object-storage/.test(n))              return "infra";
  if(/status/.test(n))                                                 return "ops";
  return "app";
}

// ── Build graph data ──────────────────────────────────────────────────────────
const nodes = Object.keys(DEPS).map(id => ({
  id, name:id, state:"healthy", tier:getTier(id), color:STATE_COLOR.healthy
}));
const links = [];
Object.entries(DEPS).forEach(([s,deps]) =>
  deps.forEach(t => links.push({source:s, target:t, rps:40+Math.random()*60}))
);

// ── Init 3d-force-graph ───────────────────────────────────────────────────────
const Graph = ForceGraph3D()(document.getElementById("graph"))
  .backgroundColor("#0a0a0c")
  .graphData({nodes, links})
  .nodeThreeObject(node => {
    const col = new THREE.Color(node.color || "#00ff88");
    const mat = new THREE.MeshStandardMaterial({
      color:col, emissive:col,
      emissiveIntensity: node.state==="healthy" ? 0.35 : 0.7,
      roughness:0.25, metalness:0.6
    });
    const mesh = new THREE.Mesh(new THREE.SphereGeometry(5,32,32), mat);
    // glow shell
    mesh.add(new THREE.Mesh(
      new THREE.SphereGeometry(7.5,32,32),
      new THREE.MeshBasicMaterial({color:col,transparent:true,opacity:0.12,side:THREE.BackSide})
    ));
    node.__mesh = mesh;
    node.__mat  = mat;
    // breathing tween
    if(node.__tween) node.__tween.kill();
    node.__tween = gsap.to(mat, {
      emissiveIntensity: (node.state==="healthy"?0.35:0.7)*2.2,
      duration: node.state==="healthy"?2.5:0.8,
      yoyo:true, repeat:-1, ease:"sine.inOut"
    });
    return mesh;
  })
  .nodeThreeObjectExtend(false)
  .nodeLabel(node => `
    <div style="background:rgba(10,10,12,.92);border:1px solid rgba(255,255,255,.1);
      border-radius:8px;padding:10px;backdrop-filter:blur(12px);font-size:12px">
      <b style="color:#fff">${node.name}</b><br>
      State: <b style="color:${STATE_COLOR[node.state]||'#fff'}">${node.state}</b><br>
      Tier: ${node.tier}
    </div>`)
  .linkColor(link => {
    const src = typeof link.source==="object" ? link.source : nodes.find(n=>n.id===link.source);
    if(src?.state==="down")     return "rgba(255,0,68,.35)";
    if(src?.state==="degraded") return "rgba(255,170,0,.3)";
    return "rgba(255,255,255,.07)";
  })
  .linkWidth(0.8)
  .linkDirectionalParticles(link => {
    const src = typeof link.source==="object" ? link.source : nodes.find(n=>n.id===link.source);
    if(src?.state==="down")     return 6;
    if(src?.state==="degraded") return 4;
    return 2;
  })
  .linkDirectionalParticleSpeed(link => {
    const src = typeof link.source==="object" ? link.source : nodes.find(n=>n.id===link.source);
    if(src?.state==="degraded") return 0.003+Math.random()*0.008;
    if(src?.state==="down")     return 0.015+Math.random()*0.01;
    return 0.002+(((link.rps||50)/100)*0.006);
  })
  .linkDirectionalParticleColor(link => {
    const src = typeof link.source==="object" ? link.source : nodes.find(n=>n.id===link.source);
    if(src?.state==="down")     return "#ff0044";
    if(src?.state==="degraded") return "#ffaa00";
    return "rgba(255,255,255,.55)";
  })
  .linkDirectionalParticleWidth(2.5)
  .onNodeClick(node => {
    const d = document.getElementById("node-detail");
    d.style.display = "block";
    document.getElementById("nd-name").textContent  = node.name;
    document.getElementById("nd-state").textContent = node.state;
    document.getElementById("nd-state").style.color = STATE_COLOR[node.state]||"#fff";
    document.getElementById("nd-tier").textContent  = node.tier;
    if(node.cpu!=null){
      document.getElementById("nd-cpu").textContent=(node.cpu*100).toFixed(1)+"%";
      document.getElementById("nd-cpu-row").style.display="block";
    } else document.getElementById("nd-cpu-row").style.display="none";
    if(node.memory!=null){
      document.getElementById("nd-mem").textContent=(node.memory*100).toFixed(1)+"%";
      document.getElementById("nd-mem-row").style.display="block";
    } else document.getElementById("nd-mem-row").style.display="none";
  })
  .enableNodeDrag(false)
  .showNavInfo(false);

// ── Cluster layout via per-tier radial nudge ──────────────────────────────────
const CENTRES = {
  data:{x:180,y:0,z:0}, app:{x:60,y:60,z:60}, edge:{x:-160,y:40,z:40},
  infra:{x:0,y:180,z:0}, observability:{x:0,y:-160,z:80}, ops:{x:-80,y:-80,z:-80}
};
Object.entries(CENTRES).forEach(([tier,c]) => {
  Graph.d3Force("radial-"+tier, () => {
    nodes.filter(n=>n.tier===tier).forEach(n => {
      if(n.x==null) return;
      n.x += (c.x-n.x)*0.18; n.y += (c.y-n.y)*0.18; n.z += (c.z-n.z)*0.18;
    });
  });
});

// ── Bloom post-processing (after engine ready) ────────────────────────────────
let composer = null;
Graph.onEngineStop(() => {
  if(!composer){
    const renderer = Graph.renderer();
    const scene    = Graph.scene();
    const camera   = Graph.camera();
    composer = new THREE.EffectComposer(renderer);
    composer.addPass(new THREE.RenderPass(scene, camera));
    const bloom = new THREE.UnrealBloomPass(
      new THREE.Vector2(window.innerWidth, window.innerHeight), 1.4, 0.6, 0.1
    );
    composer.addPass(bloom);
    Graph.pauseAnimation();
    (function loop(){ requestAnimationFrame(loop); if(composer) composer.render(); })();
  }
  // Entrance spiral
  const cam = Graph.camera();
  nodes.forEach((node,i) => {
    const mesh = node.__mesh;
    if(!mesh) return;
    const fx=mesh.position.x, fy=mesh.position.y, fz=mesh.position.z;
    const angle=(i/nodes.length)*Math.PI*4;
    mesh.position.set(cam.position.x+Math.cos(angle)*60, cam.position.y+Math.sin(angle)*60, cam.position.z-800);
    mesh.scale.set(0.01,0.01,0.01);
    gsap.to(mesh.position,{x:fx,y:fy,z:fz,duration:2,delay:i*0.025,ease:"expo.out"});
    gsap.to(mesh.scale,{x:1,y:1,z:1,duration:2,delay:i*0.025,ease:"expo.out"});
  });
});

// ── Auto-rotation ─────────────────────────────────────────────────────────────
let rotating = true;
(function rot(){
  requestAnimationFrame(rot);
  if(!rotating) return;
  const cam = Graph.camera();
  const r = Math.sqrt(cam.position.x**2+cam.position.z**2);
  const a = Math.atan2(cam.position.z, cam.position.x);
  cam.position.x = r*Math.cos(a+0.0008);
  cam.position.z = r*Math.sin(a+0.0008);
  cam.lookAt(0,0,0);
})();
document.getElementById("rotateBtn").addEventListener("click",()=>{
  rotating=!rotating;
  document.getElementById("rotateBtn").textContent=rotating?"Stop Rotation":"Start Rotation";
});

// ── State update ──────────────────────────────────────────────────────────────
let sessionId = null;

function updateGraph(services, details, rps){
  nodes.forEach(node => {
    const newState = services[node.id]||"healthy";
    const detail   = details[node.id]||{};
    const newColor = STATE_COLOR[newState]||STATE_COLOR.healthy;
    const wasDown  = node.state==="down";
    node.state  = newState;
    node.color  = newColor;
    node.cpu    = detail.cpu;
    node.memory = detail.memory;
    if(node.__mat){
      const col = new THREE.Color(newColor);
      node.__mat.color.set(col);
      node.__mat.emissive.set(col);
      if(!wasDown && newState==="down"){
        // failure flash
        gsap.to(node.__mesh.scale,{x:2,y:2,z:2,duration:.2,yoyo:true,repeat:7,ease:"power3.inOut"});
        gsap.to(node.__mat,{emissiveIntensity:3.5,duration:.2,yoyo:true,repeat:7,ease:"power3.inOut"});
      }
      if(node.__tween) node.__tween.kill();
      node.__tween = gsap.to(node.__mat,{
        emissiveIntensity:(newState==="healthy"?0.35:0.7)*2.2,
        duration:newState==="healthy"?2.5:0.8,
        yoyo:true,repeat:-1,ease:"sine.inOut"
      });
    }
  });
  // refresh link colours / particles
  Graph.graphData(Graph.graphData());
  // metrics bar
  const counts = {healthy:0,degraded:0,down:0};
  Object.values(services).forEach(s=>{ if(counts[s]!=null) counts[s]++; });
  document.getElementById("m-healthy").textContent  = counts.healthy;
  document.getElementById("m-degraded").textContent = counts.degraded;
  document.getElementById("m-down").textContent     = counts.down;
  document.getElementById("m-rps").textContent      = Math.round(rps);
}

async function loadState(){
  if(!sessionId) return;
  try{
    const r = await fetch("/state?session_id="+encodeURIComponent(sessionId));
    const d = await r.json();
    const st = d.state;
    updateGraph(st.services||{}, st.service_details||{}, st.metrics?.requests_per_sec||0);
    const m = st.metrics||{};
    document.getElementById("m-err").textContent   = (m.error_rate!=null?m.error_rate.toFixed(3):"—");
    document.getElementById("m-avail").textContent = (m.availability!=null?m.availability+"%":"—");
    const h=Object.values(st.services||{}).filter(s=>s==="healthy").length;
    const t=Object.keys(st.services||{}).length;
    document.getElementById("status").textContent =
      `${h}/${t} healthy · RPS ${Math.round(m.requests_per_sec||0)} · Tick ${st.incident?.events?.length||0}`;
  }catch(e){ document.getElementById("status").textContent="Error: "+e.message; }
}

async function resetScenario(){
  document.getElementById("status").textContent="Resetting…";
  const task = document.getElementById("scenario").value;
  const seed = document.getElementById("seed").value;
  try{
    const r = await fetch("/reset",{
      method:"POST", headers:{"Content-Type":"application/json"},
      body:JSON.stringify({task_id:task, seed:seed?Number(seed):null})
    });
    const d = await r.json();
    sessionId = d.session_id;
    document.getElementById("status").textContent="Session: "+sessionId.substring(0,12)+"…";
    await loadState();
  }catch(e){ document.getElementById("status").textContent="Error: "+e.message; }
}

document.getElementById("resetBtn").addEventListener("click",  resetScenario);
document.getElementById("refreshBtn").addEventListener("click", loadState);
setInterval(loadState, 3000);
</script>
</body>
</html>"""



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