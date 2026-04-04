"""
environment/server.py — FastAPI wrapper for DevOpsWarRoomEnv

Exposes the three required OpenEnv endpoints:
  POST /reset   — reset environment to a given task scenario
  POST /step    — execute one agent action
  GET  /state   — return current raw environment state

Plus health-check and root endpoints for HuggingFace Space monitoring.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, Optional
import traceback

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
    """Health check endpoint for HuggingFace Space monitoring."""
    return {"status": "ok"}


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
