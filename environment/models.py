from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class Role(str, Enum):
    SRE = "SRE"
    DEV = "Dev"
    MANAGER = "Manager"

class ServiceState(str, Enum):
    healthy = "healthy"
    degraded = "degraded"
    down = "down"
    restarting = "restarting"

class AvailabilityZone(str, Enum):
    us_east_1a = "us-east-1a"
    us_east_1b = "us-east-1b"
    us_east_1c = "us-east-1c"

class Scenario(str, Enum):
    Easy = "Easy"
    Medium = "Medium"
    Hard = "Hard"

# Uppercase aliases so both Scenario.EASY and Scenario.Easy work
Scenario.EASY = Scenario.Easy
Scenario.MEDIUM = Scenario.Medium
Scenario.HARD = Scenario.Hard

class Alert(BaseModel):
    severity: str
    message: str
    fired_at: int

class Observation(BaseModel):
    tick: int
    current_role: str
    services: Dict[str, ServiceState]
    alerts: List[Alert]
    metrics: Optional[Dict[str, Any]] = None

    # SRE specifically
    logs: Optional[List[str]] = None

    # Dev specifically
    deployment_history: Optional[List[str]] = None
    code_diffs: Optional[List[str]] = None

    # Manager specifically
    sla_status: Optional[str] = None
    estimated_affected_users: Optional[int] = None

    # Multi-zone topology (SRE only)
    zone_health: Optional[Dict[str, str]] = None  # zone -> "healthy" | "degraded" | "down"
    service_distribution: Optional[Dict[str, List[str]]] = None  # service -> [zones]

    # Step budget
    steps_remaining: Optional[int] = None

    # Dict-style access for SDK compatibility
    def __getitem__(self, key: str):
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key) and getattr(self, key) is not None

class Action(BaseModel):
    action_type: str
    target: Optional[str] = None
    params: Optional[Dict[str, Any]] = None

class Reward(BaseModel):
    value: float
    reason: str
    done: bool
