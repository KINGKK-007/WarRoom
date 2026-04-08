from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Role(str, Enum):
    SRE = "SRE"
    DEV = "Dev"
    MANAGER = "Manager"


class ServiceState(str, Enum):
    healthy = "healthy"
    degraded = "degraded"
    down = "down"
    restarting = "restarting"
    isolated = "isolated"


class AvailabilityZone(str, Enum):
    us_east_1a = "us-east-1a"
    us_east_1b = "us-east-1b"
    us_east_1c = "us-east-1c"
    us_central_1a = "us-central-1a"


class Scenario(str, Enum):
    Easy = "Easy"
    Medium = "Medium"
    Hard = "Hard"
    EasyRedis = "EasyRedis"
    MediumKafka = "MediumKafka"
    HardMesh = "HardMesh"
    MediumReplica = "MediumReplica"
    MediumCache = "MediumCache"
    HardRollback = "HardRollback"
    HardDNS = "HardDNS"
    HardRegion = "HardRegion"
    Chaos = "Chaos"


Scenario.EASY = Scenario.Easy
Scenario.MEDIUM = Scenario.Medium
Scenario.HARD = Scenario.Hard
Scenario.EASY_REDIS = Scenario.EasyRedis
Scenario.MEDIUM_KAFKA = Scenario.MediumKafka
Scenario.HARD_MESH = Scenario.HardMesh
Scenario.MEDIUM_REPLICA = Scenario.MediumReplica
Scenario.MEDIUM_CACHE = Scenario.MediumCache
Scenario.HARD_ROLLBACK = Scenario.HardRollback
Scenario.HARD_DNS = Scenario.HardDNS
Scenario.HARD_REGION = Scenario.HardRegion
Scenario.CHAOS = Scenario.Chaos


class Alert(BaseModel):
    severity: str
    message: str
    fired_at: int
    source: Optional[str] = None


class Observation(BaseModel):
    tick: int
    current_role: str
    services: Dict[str, ServiceState]
    alerts: List[Alert]
    metrics: Optional[Dict[str, Any]] = None
    logs: Optional[List[Any]] = None
    deployment_history: Optional[List[Any]] = None
    code_diffs: Optional[List[str]] = None
    sla_status: Optional[str] = None
    estimated_affected_users: Optional[int] = None
    zone_health: Optional[Dict[str, str]] = None
    service_distribution: Optional[Dict[str, List[str]]] = None
    traces: Optional[List[Dict[str, Any]]] = None
    metrics_history: Optional[List[Dict[str, Any]]] = None
    incident_summary: Optional[Dict[str, Any]] = None
    available_actions: List[str] = Field(default_factory=list)
    suggested_actions: List[str] = Field(default_factory=list)
    steps_remaining: Optional[int] = None

    def __getitem__(self, key: str):
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key) and getattr(self, key) is not None


class Action(BaseModel):
    action_type: str
    target: Optional[str] = None
    params: Optional[Dict[str, Any]] = None


class Reward(BaseModel):
    value: float = Field(ge=0.0, le=1.0)
    reason: str
    done: bool
