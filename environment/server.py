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
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any, Dict, Optional
import traceback
import json

from .env import DevOpsWarRoomEnv
from .models import Action, Scenario, ServiceState, Alert

# ---------------------------------------------------------------------------
# App + global env instance
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

# Single global environment instance (one env per container per OpenEnv spec)
env = DevOpsWarRoomEnv()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: Scenario = Scenario.EASY


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: Dict[str, Any]
    done: bool
    info: Dict[str, Any]


# ---------------------------------------------------------------------------
# Serialization helper
# ---------------------------------------------------------------------------

def _serialize_state(state: dict) -> dict:
    """Deep-convert a state dict so all Pydantic models, enums, and nested
    objects become plain JSON-safe Python types."""
    result = {}
    for key, value in state.items():
        if isinstance(value, dict):
            result[key] = {
                k: (v.value if isinstance(v, ServiceState) else v)
                for k, v in value.items()
            }
        elif isinstance(value, list):
            serialized_list = []
            for item in value:
                if isinstance(item, Alert):
                    serialized_list.append(
                        item.model_dump() if hasattr(item, "model_dump") else item.dict()
                    )
                else:
                    serialized_list.append(item)
            result[key] = serialized_list
        else:
            result[key] = value
    return result


def _obs_to_dict(obs) -> dict:
    """Convert an Observation Pydantic model to a plain dict."""
    if hasattr(obs, "model_dump"):
        d = obs.model_dump()
    else:
        d = obs.dict()
    # Ensure ServiceState enums in 'services' are plain strings
    if "services" in d:
        d["services"] = {
            k: (v.value if isinstance(v, ServiceState) else v)
            for k, v in d["services"].items()
        }
    return d


def _reward_to_dict(reward) -> dict:
    if hasattr(reward, "model_dump"):
        return reward.model_dump()
    return reward.dict()


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
    from .models import Action as ActionModel, Observation as ObsModel
    return {
        "action": ActionModel.model_json_schema(),
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
                        "inputSchema": {"type": "object", "properties": {"task_id": {"type": "string", "enum": ["Easy", "Medium", "Hard"]}}},
                    },
                    {
                        "name": "step",
                        "description": "Execute an action in the environment",
                        "inputSchema": {"type": "object", "properties": {"action_type": {"type": "string"}, "target": {"type": "string"}}},
                    },
                    {
                        "name": "state",
                        "description": "Get the current environment state",
                        "inputSchema": {"type": "object", "properties": {}},
                    },
                ],
            },
        })

    if method == "tools/call":
        tool_name = body.get("params", {}).get("name", "")
        tool_args = body.get("params", {}).get("arguments", {})

        try:
            if tool_name == "reset":
                task_id = Scenario(tool_args.get("task_id", "Easy"))
                obs = env.reset(task_id)
                result_data = _obs_to_dict(obs)
            elif tool_name == "step":
                action = Action(**tool_args)
                obs, reward, done, info = env.step(action)
                result_data = {"observation": _obs_to_dict(obs), "reward": _reward_to_dict(reward), "done": done, "info": info}
            elif tool_name == "state":
                result_data = _serialize_state(env.state)
            else:
                return JSONResponse({"jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}})

            return JSONResponse({"jsonrpc": "2.0", "id": rpc_id, "result": [{"type": "text", "text": json.dumps(result_data)}]})
        except Exception as e:
            return JSONResponse({"jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32000, "message": str(e)}})

    return JSONResponse({"jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32601, "message": f"Method not found: {method}"}})


@app.post("/reset")
def reset(request: ResetRequest):
    """Reset the environment against a specific task scenario.

    Returns the initial observation as a JSON dict.
    """
    try:
        obs = env.reset(request.task_id)
        return _obs_to_dict(obs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@app.post("/step")
def step(action: Action):
    """Execute one agent action in the environment.

    Accepts an Action model and returns:
      - observation: the new environment observation
      - reward: {value, reason, done}
      - done: whether the episode has ended
      - info: additional metadata
    """
    try:
        obs, reward, done, info = env.step(action)
        return {
            "observation": _obs_to_dict(obs),
            "reward": _reward_to_dict(reward),
            "done": done,
            "info": info,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Step failed: {str(e)}\n{traceback.format_exc()}",
        )


@app.get("/state")
def get_state():
    """Return the current raw environment state for debugging/inspection.

    All Pydantic models and enums are serialized to plain JSON types.
    """
    try:
        return _serialize_state(env.state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"State retrieval failed: {str(e)}")
