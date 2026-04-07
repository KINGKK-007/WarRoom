import copy
from typing import Any, Dict, List, Tuple

from .actions import ACTION_CATALOG, CommandParser
from .models import Action, Alert, Observation, Reward, Role, Scenario, ServiceState
from .roles import RoleManager
from .scenarios import SCENARIOS, SERVICE_DEPENDENCIES, generate_procedural_incident, generate_variant

try:
    from openenv import BaseEnvironment
except ImportError:
    class BaseEnvironment:
        pass


class DevOpsWarRoomEnv(BaseEnvironment):
    max_steps = 24

    def __init__(self, role: Role = Role.SRE):
        self.role_manager = RoleManager(role)
        self.role = self.role_manager.current_role
        self.parser = CommandParser()
        self._state: Dict[str, Any] = {}
        self.history: Dict[str, Any] = {}
        self.tick_count = 0
        self.step_count = 0
        self.task_id = Scenario.EASY
        self.action_history: List[Dict[str, Any]] = []
        self.reset(self.task_id)

    @property
    def state(self) -> dict:
        return self._state

    def timeline(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(self._state.get("incident", {}).get("events", []))

    def reset(self, task_id: Scenario = Scenario.EASY, seed: int | None = None) -> Observation:
        self.step_count = 0
        self.tick_count = 0
        self.task_id = task_id
        self.role_manager = RoleManager(Role.SRE)
        self.role = self.role_manager.current_role
        if seed is not None:
            self._state = generate_variant(task_id, seed)
        elif task_id == Scenario.CHAOS:
            self._state = generate_procedural_incident(seed=20260407)
        else:
            self._state = copy.deepcopy(SCENARIOS[task_id])
        self.history = {"seen_evidence": set(), "executed_actions": set()}
        self.action_history = []
        self._refresh_service_states()
        self._sync_sla_status()
        self._record_metrics_snapshot("reset")
        return self._get_observation()

    def step(self, action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if isinstance(action, dict):
            action = Action(**action)
        self.step_count += 1

        action_name, target, params = self._normalize_action(action)
        if action_name == "unknown":
            self.tick("unknown")
            reward = Reward(value=-0.12 - self._ignored_alert_penalty(), reason="Unrecognized command.", done=self._is_done())
            return self._get_observation(), reward, self._is_done(), {"error": "unknown_action"}

        if action_name == "switch_role":
            self.tick("switch_role")
            success = target is not None and self.role_manager.issue_switch_role(target)
            if success:
                self.role = self.role_manager.current_role
                self._record_action(action_name, target, True)
                reward = Reward(value=max(-1.0, min(1.0, 0.03 - self._ignored_alert_penalty())), reason=f"Role switched to {self.role.value}.", done=self._is_done())
                return self._get_observation(), reward, self._is_done(), {"role": self.role.value}
            reward = Reward(value=-0.06, reason="Invalid role.", done=self._is_done())
            return self._get_observation(), reward, self._is_done(), {"error": "invalid_role"}

        if not self.role_manager.check_action_allowed(action_name):
            reward = Reward(value=-0.18, reason="Unauthorized action for current role.", done=self._is_done())
            return self._get_observation(), reward, self._is_done(), {"error": "unauthorized"}

        handler_map = {
            "inspect": lambda: self._inspect(target),
            "query_metrics": self._query_metrics,
            "query_logs": self._query_logs,
            "query_traces": self._query_traces,
            "query_deploy": self._query_deploy,
            "query_topology": self._query_topology,
            "run_health_check": lambda: self._run_health_check(target),
            "restart_service": lambda: self._restart_service(target),
            "rollback_deploy": lambda: self._rollback_deploy(target, params.get("version")),
            "scale": lambda: self._scale_service(target, params.get("count")),
            "clear_queue": lambda: self._clear_queue(target),
            "tune_autoscaling": lambda: self._tune_autoscaling(target),
            "failover_zone": lambda: self._failover_zone(target),
            "drain_zone": lambda: self._drain_zone(target),
            "restore_zone": lambda: self._restore_zone(target),
            "rebalance_traffic": lambda: self._rebalance_traffic(target),
            "throttle_service": lambda: self._throttle_service(target),
            "isolate_service": lambda: self._isolate_service(target),
            "acknowledge_alert": lambda: self._acknowledge_alert(target),
            "escalate": lambda: self._escalate(target),
            "notify": lambda: self._notify(target),
            "update_status_page": lambda: self._update_status_page(target),
            "attach_runbook": lambda: self._attach_runbook(target),
            "verify_sla": self._verify_sla,
            "run_rca": self._run_rca,
            "generate_postmortem": self._generate_postmortem,
        }
        success, reward_value, reason, info = handler_map[action_name]()

        self._record_action(action_name, target, success)
        self.tick(action_name)

        if success:
            reward_value += self._reward_for_action(action_name, target)
        else:
            reward_value -= 0.08
        reward_value -= self._ignored_alert_penalty(action_name)

        self._refresh_service_states()
        self._sync_sla_status()
        self._update_incident_resolution()
        reward = Reward(value=max(-1.0, min(1.0, reward_value)), reason=reason, done=self._is_done())
        return self._get_observation(), reward, self._is_done(), info

    def _normalize_action(self, action: Action):
        params = copy.deepcopy(action.params or {})
        if action.action_type in {"command", "raw_command"}:
            raw_command = action.target or params.get("command", "")
            parsed = self.parser.parse(str(raw_command))
            action_name = parsed.get("action", "unknown")
            target = parsed.get("target")
            params.update(parsed)
        else:
            action_name = action.action_type
            target = action.target
        if "count" in params and params["count"] is not None:
            try:
                params["count"] = int(params["count"])
            except (TypeError, ValueError):
                params["count"] = None
        return action_name, target, params

    def _record_action(self, action_name: str, target: str, success: bool):
        entry = {"tick": self.tick_count, "action": action_name, "target": target, "success": success}
        self.action_history.append(entry)
        self._state["incident"]["actions_executed"].append(entry)
        self.history["executed_actions"].add(f"{action_name}:{target}" if target else action_name)
        self._state["incident"].setdefault("events", []).append(
            {
                "tick": self.tick_count,
                "type": "action",
                "summary": f"{action_name} executed{' successfully' if success else ' unsuccessfully'}",
                "payload": {"target": target, "success": success},
            }
        )

    def _mark_evidence(self, evidence_key: str):
        self.history["seen_evidence"].add(evidence_key)
        if evidence_key not in self._state["incident"]["evidence_collected"]:
            self._state["incident"]["evidence_collected"].append(evidence_key)

    def _mark_mitigation(self, mitigation_key: str):
        if mitigation_key not in self._state["incident"]["mitigations_applied"]:
            self._state["incident"]["mitigations_applied"].append(mitigation_key)

    def _mark_production_action(self, action_key: str):
        if action_key not in self._state["incident"]["production_actions_completed"]:
            self._state["incident"]["production_actions_completed"].append(action_key)

    def _service_detail(self, target: str):
        return self._state["service_details"].get(target)

    def _inspect(self, target: str):
        detail = self._service_detail(target) if target else None
        if detail is None:
            return False, -0.02, "Unknown service.", {"error": "unknown_service"}
        self._mark_evidence(f"inspect:{target}")
        data = copy.deepcopy(detail)
        data["status"] = self._state["services"][target].value
        reward = 0.07 if target in self._state["incident"]["root_causes"] or target in self._state["incident"]["service_targets"] else 0.02
        return True, reward, f"Inspected {target}.", {"service": target, "detail": data}

    def _query_metrics(self):
        self._mark_evidence("query_metrics")
        return True, 0.05, "Queried metrics.", {"metrics": copy.deepcopy(self._state["metrics"]), "history": copy.deepcopy(self._state["metrics_history"][-6:])}

    def _query_logs(self):
        self._mark_evidence("query_logs")
        return True, 0.04, "Queried logs.", {"logs": copy.deepcopy(self._state["logs"][-20:])}

    def _query_traces(self):
        self._mark_evidence("query_traces")
        return True, 0.05, "Queried traces.", {"traces": copy.deepcopy(self._state["traces"][-12:])}

    def _query_deploy(self):
        self._mark_evidence("query_deploy")
        return True, 0.06, "Queried deployment history.", {"deploy_history": copy.deepcopy(self._state["deploy_history"][-12:]), "code_diffs": copy.deepcopy(self._state["code_diffs"])}

    def _query_topology(self):
        self._mark_evidence("query_topology")
        return True, 0.05, "Queried topology.", {"zones": copy.deepcopy(self._state["zones"]), "service_distribution": copy.deepcopy(self._state["service_distribution"])}

    def _run_health_check(self, target: str):
        target = target or (self._state["incident"]["service_targets"][0] if self._state["incident"]["service_targets"] else None)
        detail = self._service_detail(target) if target else None
        if detail is None:
            return False, -0.03, "Unknown health check target.", {"error": "unknown_service"}
        detail["health_checks"] += 1
        self._mark_evidence(f"run_health_check:{target}")
        return True, 0.05, f"Ran health check for {target}.", {"health": self._state["services"][target].value}

    def _restart_service(self, target: str):
        detail = self._service_detail(target) if target else None
        if detail is None:
            return False, -0.1, "Unknown service restart target.", {"error": "unknown_service"}
        was_healthy = self._state["services"][target] == ServiceState.healthy
        detail["isolated"] = False
        detail["queue_depth"] = max(0, detail["queue_depth"] // 3)
        detail["cpu"] = max(0.25, detail["cpu"] - 0.2)
        detail["memory"] = max(0.22, detail["memory"] - 0.12)
        for zone in detail["zone_states"]:
            if self._state["zones"][zone]["status"] != "down":
                detail["zone_states"][zone] = ServiceState.healthy
        self._mark_mitigation(f"restart_service:{target}")
        if was_healthy:
            self._state["incident"]["unnecessary_restarts"].append(target)
        if target == "postgres-primary":
            self._state["service_details"]["postgres-replica"]["zone_states"] = copy.deepcopy(detail["zone_states"])
        return True, (0.16 if not was_healthy else -0.12), f"Restarted {target}.", {"service": target, "was_healthy": was_healthy}

    def _rollback_deploy(self, target: str, version: str):
        detail = self._service_detail(target) if target else None
        if detail is None:
            return False, -0.1, "Unknown rollback target.", {"error": "unknown_service"}
        expected = self._state["incident"]["deploy_targets"].get(target)
        if expected is None:
            return False, -0.08, "No rollback required for this service.", {"service": target}
        applied_version = version or expected
        detail["version"] = applied_version
        detail["cpu"] = max(0.24, detail["cpu"] - 0.18)
        detail["memory"] = max(0.24, detail["memory"] - 0.1)
        for zone in detail["zone_states"]:
            if self._state["zones"][zone]["status"] not in {"down"}:
                detail["zone_states"][zone] = ServiceState.healthy
        self._state["code_diffs"].append(f"Rollback applied on {target} to {applied_version}")
        self._mark_mitigation(f"rollback_deploy:{target}")
        return True, (0.18 if applied_version == expected else 0.04), f"Rolled back {target} to {applied_version}.", {"service": target, "version": applied_version}

    def _scale_service(self, target: str, count: int):
        detail = self._service_detail(target) if target else None
        if detail is None:
            return False, -0.08, "Unknown scale target.", {"error": "unknown_service"}
        add = max(1, count or 2)
        detail["replicas"] += add
        detail["cpu"] = max(0.2, detail["cpu"] - 0.08 * min(add, 4))
        detail["memory"] = max(0.22, detail["memory"] - 0.05 * min(add, 4))
        detail["queue_depth"] = max(0, detail["queue_depth"] - 2500 * add)
        self._mark_mitigation(f"scale:{target}")
        return True, 0.1, f"Scaled {target}.", {"replicas": detail["replicas"]}

    def _clear_queue(self, target: str):
        detail = self._service_detail(target) if target else None
        if detail is None:
            return False, -0.08, "Unknown queue target.", {"error": "unknown_service"}
        detail["queue_depth"] = max(0, detail["queue_depth"] - 25000)
        detail["memory"] = max(0.24, detail["memory"] - 0.08)
        detail["cpu"] = max(0.24, detail["cpu"] - 0.05)
        if detail["queue_depth"] <= 4000:
            for zone in detail["zone_states"]:
                if self._state["zones"][zone]["status"] != "down":
                    detail["zone_states"][zone] = ServiceState.healthy
        self._mark_mitigation(f"clear_queue:{target}")
        return True, 0.13, f"Cleared queue for {target}.", {"queue_depth": detail["queue_depth"]}

    def _tune_autoscaling(self, target: str):
        detail = self._service_detail(target) if target else None
        if detail is None:
            return False, -0.06, "Unknown autoscaling target.", {"error": "unknown_service"}
        detail["autoscaling_tuned"] = True
        detail["cpu"] = max(0.18, detail["cpu"] - 0.1)
        detail["memory"] = max(0.22, detail["memory"] - 0.06)
        if detail["queue_depth"] <= 5000:
            for zone in detail["zone_states"]:
                if self._state["zones"][zone]["status"] != "down":
                    detail["zone_states"][zone] = ServiceState.healthy
        self._mark_mitigation(f"tune_autoscaling:{target}")
        return True, 0.11, f"Tuned autoscaling for {target}.", {"autoscaling_tuned": True}

    def _failover_zone(self, target: str):
        zone = self._state["zones"].get(target)
        if zone is None:
            return False, -0.08, "Unknown zone.", {"error": "unknown_zone"}
        zone["status"] = "drained"
        zone["drained"] = True
        zone["failed_over"] = True
        zone["packet_loss"] = 0.0
        zone["latency_ms"] = 18
        for service, detail in self._state["service_details"].items():
            if target in detail["zone_states"]:
                detail["zone_states"][target] = ServiceState.healthy
                if detail["replicas"] < len(detail["zones"]) + 1 and service in self._state["incident"]["service_targets"]:
                    detail["replicas"] += 1
        self._mark_mitigation(f"failover_zone:{target}")
        return True, 0.15, f"Failed over zone {target}.", {"zone": target}

    def _drain_zone(self, target: str):
        zone = self._state["zones"].get(target)
        if zone is None:
            return False, -0.06, "Unknown zone.", {"error": "unknown_zone"}
        zone["status"] = "drained"
        zone["drained"] = True
        zone["packet_loss"] = max(0.0, zone["packet_loss"] - 0.1)
        zone["latency_ms"] = max(20, zone["latency_ms"] - 20)
        for detail in self._state["service_details"].values():
            if target in detail["zone_states"] and detail["zone_states"][target] == ServiceState.down:
                detail["zone_states"][target] = ServiceState.degraded
        self._mark_mitigation(f"drain_zone:{target}")
        return True, 0.08, f"Drained zone {target}.", {"zone": target}

    def _restore_zone(self, target: str):
        zone = self._state["zones"].get(target)
        if zone is None:
            return False, -0.06, "Unknown zone.", {"error": "unknown_zone"}
        zone["status"] = "healthy"
        zone["drained"] = False
        zone["packet_loss"] = 0.0
        zone["latency_ms"] = 12
        for detail in self._state["service_details"].values():
            if target in detail["zone_states"] and detail["zone_states"][target] != ServiceState.isolated:
                detail["zone_states"][target] = ServiceState.healthy
        if target == "us-east-1b" and self._state["service_details"]["worker-service"]["queue_depth"] <= 5000:
            for service in ["worker-service", "scheduler-service", "notification-service", "analytics-service"]:
                for zone_name in self._state["service_details"][service]["zone_states"]:
                    if self._state["zones"][zone_name]["status"] == "healthy":
                        self._state["service_details"][service]["zone_states"][zone_name] = ServiceState.healthy
        self._mark_mitigation(f"restore_zone:{target}")
        return True, 0.09, f"Restored zone {target}.", {"zone": target}

    def _rebalance_traffic(self, target: str):
        detail = self._service_detail(target) if target else None
        if detail is None:
            return False, -0.06, "Unknown traffic rebalance target.", {"error": "unknown_service"}
        detail["traffic_weight"] = 0.65
        detail["cpu"] = max(0.2, detail["cpu"] - 0.08)
        for zone in detail["zone_states"]:
            if self._state["zones"][zone]["status"] == "healthy":
                detail["zone_states"][zone] = ServiceState.healthy
        self._mark_mitigation(f"rebalance_traffic:{target}")
        return True, 0.12, f"Rebalanced traffic for {target}.", {"traffic_weight": detail["traffic_weight"]}

    def _throttle_service(self, target: str):
        detail = self._service_detail(target) if target else None
        if detail is None:
            return False, -0.06, "Unknown throttle target.", {"error": "unknown_service"}
        detail["traffic_weight"] = max(0.25, detail["traffic_weight"] - 0.25)
        detail["cpu"] = max(0.2, detail["cpu"] - 0.04)
        detail["memory"] = max(0.2, detail["memory"] - 0.03)
        return True, 0.05, f"Throttled {target}.", {"service": target}

    def _isolate_service(self, target: str):
        detail = self._service_detail(target) if target else None
        if detail is None:
            return False, -0.06, "Unknown isolate target.", {"error": "unknown_service"}
        detail["isolated"] = True
        for zone in detail["zone_states"]:
            detail["zone_states"][zone] = ServiceState.isolated
        return True, 0.05, f"Isolated {target}.", {"service": target}

    def _acknowledge_alert(self, target: str):
        self._state["incident"]["acknowledged_alerts"] = True
        self._mark_production_action("acknowledge_alert")
        return True, 0.04, "Acknowledged active alerts.", {"alert": target or "all"}

    def _escalate(self, target: str):
        target = target or "incident-commander"
        if target not in self._state["incident"]["escalated_to"]:
            self._state["incident"]["escalated_to"].append(target)
        self._mark_production_action("escalate")
        return True, 0.05, f"Escalated to {target}.", {"target": target}

    def _notify(self, target: str):
        target = target or "stakeholders"
        if target not in self._state["incident"]["notified"]:
            self._state["incident"]["notified"].append(target)
        self._mark_production_action("notify")
        return True, 0.05, f"Notified {target}.", {"target": target}

    def _update_status_page(self, target: str):
        self._state["incident"]["status_page_updated"] = True
        self._mark_production_action("update_status_page")
        return True, 0.06, "Updated status page.", {"component": target or "all"}

    def _attach_runbook(self, target: str):
        self._state["incident"]["runbook_attached"] = True
        self._state["incident"]["artifact_status"]["runbook_attached"] = True
        self._mark_production_action("attach_runbook")
        return True, 0.05, "Attached runbook.", {"artifact": target or f"runbook:{self._state['incident']['name']}"}

    def _verify_sla(self):
        self._sync_sla_status(force_record=True)
        self._mark_production_action("verify_sla")
        self._state["incident"]["artifact_status"]["sla_verified"] = True
        status = self._state["sla"]["current_status"]
        return True, (0.08 if status == "CURRENTLY_MET" else -0.02), "Verified SLA.", {"sla": copy.deepcopy(self._state["sla"])}

    def _build_causal_chain(self):
        chain = []
        for root in self._state["incident"]["root_causes"]:
            impacted = [
                service
                for service, dependencies in SERVICE_DEPENDENCIES.items()
                if root in dependencies and self._state["services"].get(service) != ServiceState.healthy
            ]
            chain.append({"root": root, "downstream_impacts": impacted})
        return chain

    def _run_rca(self):
        evidence_ready = all(item in self._state["incident"]["evidence_collected"] for item in self._state["incident"]["required_evidence"][: max(2, min(5, len(self._state["incident"]["required_evidence"])) )])
        mitigation_ready = len(self._state["incident"]["mitigations_applied"]) > 0
        if not (evidence_ready and mitigation_ready):
            return False, -0.05, "Insufficient evidence for RCA.", {"error": "insufficient_evidence"}
        self._state["incident"]["rca"] = {
            "root_causes": copy.deepcopy(self._state["incident"]["root_causes"]),
            "evidence": copy.deepcopy(self._state["incident"]["evidence_collected"]),
            "mitigations": copy.deepcopy(self._state["incident"]["mitigations_applied"]),
            "ignored_alert_ticks": self._state["incident"]["ignored_alert_ticks"],
            "blast_radius": sorted(
                service for service, state in self._state["services"].items() if state != ServiceState.healthy
            ),
            "contributing_factors": [
                f"zone:{zone}" for zone, detail in self._state["zones"].items() if detail["status"] != "healthy"
            ] + [
                f"deploy:{service}" for service in self._state["incident"]["deploy_targets"]
            ],
            "ruled_out_causes": [
                service for service in ["prometheus", "grafana", "status-page"] if service not in self._state["incident"]["root_causes"]
            ],
            "recommended_followups": ["add canary verification", "extend alert coverage", "update rollback runbook"],
            "causal_chain": self._build_causal_chain(),
        }
        self._state["incident"]["artifact_status"]["rca"] = True
        self._mark_production_action("run_rca")
        return True, 0.09, "Generated RCA.", {"rca": copy.deepcopy(self._state["incident"]["rca"])}

    def _generate_postmortem(self):
        if not self._state["incident"]["artifact_status"]["rca"]:
            return False, -0.05, "RCA required before postmortem.", {"error": "rca_required"}
        self._state["incident"]["postmortem"] = {
            "incident": self._state["incident"]["name"],
            "summary": self._state["incident"]["summary"],
            "timeline": copy.deepcopy(self._state["incident"]["timeline"]),
            "follow_ups": ["runbook update", "capacity review", "chaos replay"],
            "sla_status": self._state["sla"]["current_status"],
            "rca_summary": copy.deepcopy(self._state["incident"]["rca"]),
        }
        self._state["incident"]["artifact_status"]["postmortem"] = True
        self._mark_production_action("generate_postmortem")
        return True, 0.09, "Generated postmortem.", {"postmortem": copy.deepcopy(self._state["incident"]["postmortem"])}

    def _reward_for_action(self, action_name: str, target: str) -> float:
        incident = self._state["incident"]
        bonus = 0.0
        action_key = f"{action_name}:{target}" if target else action_name
        if action_key in incident["required_evidence"] or action_name in incident["required_evidence"]:
            bonus += 0.04
        if action_key in incident["required_mitigations"]:
            bonus += 0.11
        if action_name in incident["required_production_actions"]:
            bonus += 0.06
        if action_name in {"acknowledge_alert", "verify_sla"} and incident["severity"] in {"sev1", "sev0"}:
            bonus += 0.03
        return bonus

    def _ignored_alert_penalty(self, action_name: str = "") -> float:
        critical_alerts = sum(1 for alert in self._state["alerts"] if alert.severity == "CRITICAL")
        if critical_alerts == 0:
            return 0.0
        if action_name in {"acknowledge_alert", "inspect", "query_metrics", "query_logs", "query_traces", "query_topology", "run_health_check"}:
            return 0.0
        if not self._state["incident"]["acknowledged_alerts"]:
            return min(0.15, 0.03 * critical_alerts)
        return 0.0

    def _refresh_service_states(self):
        services = self._state["services"]
        details = self._state["service_details"]
        for service, detail in details.items():
            zone_states = []
            for zone, current in detail["zone_states"].items():
                zone_status = self._state["zones"].get(zone, {}).get("status", "healthy")
                if current == ServiceState.isolated or detail["isolated"]:
                    zone_states.append(ServiceState.isolated)
                elif zone_status == "down":
                    zone_states.append(ServiceState.down)
                elif zone_status == "degraded" and current == ServiceState.healthy:
                    zone_states.append(ServiceState.degraded)
                else:
                    zone_states.append(current)
            if all(status in {ServiceState.down, ServiceState.isolated} for status in zone_states):
                services[service] = ServiceState.down if any(status == ServiceState.down for status in zone_states) else ServiceState.isolated
            elif any(status in {ServiceState.degraded, ServiceState.down, ServiceState.isolated} for status in zone_states):
                services[service] = ServiceState.degraded
            else:
                services[service] = ServiceState.healthy

        changed = True
        while changed:
            changed = False
            for service, dependencies in SERVICE_DEPENDENCIES.items():
                if not dependencies:
                    continue
                dependency_states = [services[dep] for dep in dependencies if dep in services]
                if any(state in {ServiceState.down, ServiceState.isolated} for state in dependency_states):
                    if services[service] == ServiceState.healthy:
                        services[service] = ServiceState.degraded
                        changed = True
                elif sum(1 for state in dependency_states if state == ServiceState.degraded) >= 2:
                    if services[service] == ServiceState.healthy:
                        services[service] = ServiceState.degraded
                        changed = True

        self._recompute_metrics()

    def _recompute_metrics(self):
        services = self._state["services"]
        details = self._state["service_details"]
        down = sum(1 for status in services.values() if status == ServiceState.down)
        degraded = sum(1 for status in services.values() if status == ServiceState.degraded)
        isolated = sum(1 for status in services.values() if status == ServiceState.isolated)
        queue_depth = sum(detail.get("queue_depth", 0) for detail in details.values())
        bad_zones = sum(1 for zone in self._state["zones"].values() if zone["status"] in {"degraded", "down"})
        avg_zone_latency = sum(zone["latency_ms"] for zone in self._state["zones"].values()) / max(1, len(self._state["zones"]))
        avg_packet_loss = sum(zone["packet_loss"] for zone in self._state["zones"].values())
        avg_cpu = sum(detail["cpu"] for detail in details.values()) / max(1, len(details))
        avg_memory = sum(detail["memory"] for detail in details.values()) / max(1, len(details))
        cpu = min(0.99, avg_cpu + degraded * 0.004 + down * 0.008 + bad_zones * 0.03 + min(queue_depth / 200000.0, 0.12))
        memory = min(0.99, avg_memory + degraded * 0.003 + down * 0.005 + min(queue_depth / 260000.0, 0.09))
        error_rate = min(0.99, 0.008 + degraded * 0.007 + down * 0.02 + isolated * 0.01 + min(queue_depth / 400000.0, 0.16) + bad_zones * 0.03 + avg_packet_loss * 0.2)
        latency = int(160 + degraded * 24 + down * 90 + isolated * 55 + avg_zone_latency * 0.8 + min(queue_depth / 70, 1400))
        if self._state["service_details"]["api-gateway"]["version"] == "v3.2.1":
            latency += 650
            error_rate = min(0.99, error_rate + 0.05)
        if self._state["service_details"]["api-gateway"]["traffic_weight"] < 1.0:
            latency = max(180, int(latency - 120))
        availability = max(92.0, 100.0 - down * 1.1 - degraded * 0.22 - isolated * 0.4 - bad_zones * 0.6)
        requests = max(180, int(1600 - degraded * 22 - down * 70 - isolated * 45 - min(queue_depth / 55, 500)))
        self._state["metrics"].update(
            {
                "cpu": round(cpu, 3),
                "memory": round(memory, 3),
                "error_rate": round(error_rate, 3),
                "p99_latency_ms": latency,
                "requests_per_sec": requests,
                "queue_depth": queue_depth,
                "availability": round(availability, 2),
                "saturation": round(min(0.99, (cpu + memory) / 2), 3),
            }
        )
        self._state["estimated_affected_users"] = int((100.0 - availability) * 1200 + error_rate * 10000)

    def tick(self, action_name: str):
        self.tick_count += 1
        self._advance_failure_dynamics(action_name)
        self._refresh_service_states()
        self._sync_sla_status()
        self._update_alerts()
        self._append_logs_and_traces(action_name)
        self._record_metrics_snapshot(action_name)
        self._state["incident"]["timeline"].append(
            f"tick={self.tick_count} action={action_name} cpu={self._state['metrics']['cpu']} memory={self._state['metrics']['memory']} latency={self._state['metrics']['p99_latency_ms']} error_rate={self._state['metrics']['error_rate']}"
        )
        self._state["incident"].setdefault("events", []).append(
            {
                "tick": self.tick_count,
                "type": "tick",
                "summary": f"tick {self.tick_count} after {action_name}",
                "payload": {
                    "cpu": self._state["metrics"]["cpu"],
                    "memory": self._state["metrics"]["memory"],
                    "latency": self._state["metrics"]["p99_latency_ms"],
                    "error_rate": self._state["metrics"]["error_rate"],
                },
            }
        )

    def _advance_failure_dynamics(self, action_name: str):
        incident = self._state["incident"]
        for event in list(incident.get("pending_failures", [])):
            if self.tick_count >= event["trigger_tick"]:
                if event["kind"] == "worker_oom" and "clear_queue:worker-service" not in incident["mitigations_applied"]:
                    worker = self._state["service_details"]["worker-service"]
                    for zone in worker["zone_states"]:
                        worker["zone_states"][zone] = ServiceState.down
                    self._state["service_details"]["scheduler-service"]["zone_states"]["us-east-1b"] = ServiceState.degraded
                    self._state["service_details"]["notification-service"]["zone_states"]["us-east-1b"] = ServiceState.degraded
                    self._state["logs"].append({"tick": self.tick_count, "service": "worker-service", "level": "ERROR", "message": "worker OOM event fired"})
                    self._state["incident"].setdefault("events", []).append(
                        {
                            "tick": self.tick_count,
                            "type": "failure",
                            "summary": "worker-service OOM event propagated to dependent services",
                            "payload": {"service": "worker-service"},
                        }
                    )
                incident["pending_failures"].remove(event)

        for zone_name, zone in self._state["zones"].items():
            if zone["status"] == "degraded":
                zone["packet_loss"] = min(0.4, round(zone["packet_loss"] + 0.01, 2))
                zone["latency_ms"] = min(260, zone["latency_ms"] + 8)
            elif zone["status"] == "down":
                zone["packet_loss"] = min(0.8, round(zone["packet_loss"] + 0.02, 2))
                zone["latency_ms"] = min(480, zone["latency_ms"] + 12)
            elif zone["status"] == "drained":
                zone["latency_ms"] = max(16, zone["latency_ms"] - 6)

        for service, detail in self._state["service_details"].items():
            if detail["autoscaling_tuned"]:
                detail["cpu"] = max(0.18, detail["cpu"] - 0.01)
            if service == "worker-service" and "clear_queue:worker-service" not in incident["mitigations_applied"] and incident["name"] in {"queue-backlog-and-zone-degradation", "chaos-multi-root-cause"}:
                detail["queue_depth"] += 2200
                detail["cpu"] = min(0.99, detail["cpu"] + 0.03)
                detail["memory"] = min(0.99, detail["memory"] + 0.02)
            if service == "worker-service" and detail["queue_depth"] <= 5000 and "tune_autoscaling:worker-service" in incident["mitigations_applied"]:
                for zone in detail["zone_states"]:
                    if self._state["zones"][zone]["status"] == "healthy":
                        detail["zone_states"][zone] = ServiceState.healthy
            if self._state["service_details"]["api-gateway"]["version"] == "v3.2.1" and service in {"api-gateway", "frontend-web", "mobile-bff"}:
                detail["cpu"] = min(0.99, detail["cpu"] + 0.02)
            if any(self._state["zones"][zone]["status"] == "down" for zone in detail["zones"]):
                detail["cpu"] = min(0.99, detail["cpu"] + 0.01)
                detail["memory"] = min(0.99, detail["memory"] + 0.01)

        if any(alert.severity == "CRITICAL" for alert in self._state["alerts"]) and action_name not in {"acknowledge_alert", "inspect", "query_metrics", "query_logs", "query_traces", "query_topology", "run_health_check"} and not incident["acknowledged_alerts"]:
            incident["ignored_alert_ticks"] += 1

    def _sync_sla_status(self, force_record: bool = False):
        metrics = self._state["metrics"]
        sla = self._state["sla"]
        breached = metrics["availability"] < sla["target_availability"] or metrics["error_rate"] > sla["target_error_rate"]
        sla["current_status"] = "BREACHED" if breached else "CURRENTLY_MET"
        if breached and (force_record or not sla["breaches"] or sla["breaches"][-1]["tick"] != self.tick_count):
            sla["breaches"].append({"tick": self.tick_count, "error_rate": metrics["error_rate"], "availability": metrics["availability"]})
            self._state["incident"].setdefault("events", []).append(
                {
                    "tick": self.tick_count,
                    "type": "sla_breach",
                    "summary": "SLA breach recorded",
                    "payload": {"error_rate": metrics["error_rate"], "availability": metrics["availability"]},
                }
            )

    def _update_alerts(self):
        alerts = self._state["alerts"]
        recent_sources = {alert.source for alert in alerts[-8:] if alert.source}
        for service, status in self._state["services"].items():
            if status in {ServiceState.down, ServiceState.degraded} and service not in recent_sources:
                severity = "CRITICAL" if status == ServiceState.down else "WARNING"
                alerts.append(Alert(severity=severity, message=f"{service} is {status.value}", fired_at=self.tick_count, source=service))
        for zone_name, zone in self._state["zones"].items():
            if zone["status"] in {"degraded", "down"} and zone_name not in recent_sources:
                alerts.append(Alert(severity="CRITICAL", message=f"Zone impairment detected in {zone_name}", fired_at=self.tick_count, source=zone_name))
        self._state["alerts"] = alerts[-24:]

    def _append_logs_and_traces(self, action_name: str):
        self._state["logs"].append({"tick": self.tick_count, "service": "incident-controller", "level": "INFO", "message": f"processed action {action_name}"})
        self._state["historical_logs"].append(copy.deepcopy(self._state["logs"][-1]))
        self._state["traces"].append(
            {
                "trace_id": f"trc-{self.tick_count:03d}",
                "service": "api-gateway",
                "span": action_name,
                "latency_ms": self._state["metrics"]["p99_latency_ms"],
                "status": "ok" if self._state["incident"]["resolved"] else "degraded",
                "zone": next(iter(self._state["zones"])),
            }
        )
        self._state["traces"] = self._state["traces"][-30:]

    def _record_metrics_snapshot(self, label: str):
        self._state["metrics_history"].append(
            {
                "tick": self.tick_count,
                "label": label,
                "error_rate": self._state["metrics"]["error_rate"],
                "cpu": self._state["metrics"]["cpu"],
                "memory": self._state["metrics"]["memory"],
                "p99_latency_ms": self._state["metrics"]["p99_latency_ms"],
                "availability": self._state["metrics"]["availability"],
                "queue_depth": self._state["metrics"]["queue_depth"],
            }
        )
        self._state["metrics_history"] = self._state["metrics_history"][-40:]

    def _update_incident_resolution(self):
        incident = self._state["incident"]
        metrics = self._state["metrics"]
        required_mitigations_done = all(item in incident["mitigations_applied"] for item in incident["required_mitigations"])
        production_ready = all(item in incident["production_actions_completed"] for item in {"verify_sla", "attach_runbook", "run_rca", "generate_postmortem"})
        service_targets_healthy = all(self._state["services"].get(service, ServiceState.healthy) == ServiceState.healthy for service in incident["service_targets"])
        zone_targets_healthy = all(self._state["zones"].get(zone, {}).get("status") == "healthy" for zone in incident["zone_targets"])
        no_dependency_cascade = all(status == ServiceState.healthy for status in self._state["services"].values() if status != ServiceState.isolated)
        incident["resolved"] = required_mitigations_done and service_targets_healthy and zone_targets_healthy and metrics["error_rate"] <= 0.05 and metrics["p99_latency_ms"] <= 600 and no_dependency_cascade
        if incident["resolved"] and production_ready and self._state["sla"]["current_status"] == "CURRENTLY_MET" and incident["recovery_tick"] is None:
            incident["recovery_tick"] = self.tick_count

    def _is_done(self) -> bool:
        sla_breach_terminal = len(self._state["sla"]["breaches"]) >= 3 or (
            self.step_count >= 3 and self._state["metrics"]["availability"] < 95.0
        )
        full_recovery = self._state["incident"]["recovery_tick"] is not None
        return self.step_count >= self.max_steps or full_recovery or sla_breach_terminal or self._state["metrics"]["error_rate"] >= 0.99

    def _get_observation(self) -> Observation:
        incident_summary = {
            "name": self._state["incident"]["name"],
            "severity": self._state["incident"]["severity"],
            "resolved": self._state["incident"]["resolved"],
            "evidence_collected": copy.deepcopy(self._state["incident"]["evidence_collected"]),
            "mitigations_applied": copy.deepcopy(self._state["incident"]["mitigations_applied"]),
            "production_actions_completed": copy.deepcopy(self._state["incident"]["production_actions_completed"]),
            "ignored_alert_ticks": self._state["incident"]["ignored_alert_ticks"],
            "variant": copy.deepcopy(self._state["incident"].get("variant")),
        }
        obs_data = {
            "tick": self.tick_count,
            "current_role": self.role.value,
            "services": copy.deepcopy(self._state["services"]),
            "alerts": copy.deepcopy(self._state["alerts"]),
            "zone_health": {zone: detail["status"] for zone, detail in self._state["zones"].items()},
            "service_distribution": copy.deepcopy(self._state["service_distribution"]),
            "incident_summary": incident_summary,
            "available_actions": self.role_manager.available_actions(),
            "steps_remaining": max(0, self.max_steps - self.step_count),
        }
        if self.role == Role.SRE:
            obs_data["metrics"] = copy.deepcopy(self._state["metrics"])
            obs_data["logs"] = copy.deepcopy(self._state["logs"][-12:])
            obs_data["traces"] = copy.deepcopy(self._state["traces"][-8:])
            obs_data["metrics_history"] = copy.deepcopy(self._state["metrics_history"][-8:])
        elif self.role == Role.DEV:
            obs_data["deployment_history"] = copy.deepcopy(self._state["deploy_history"][-10:])
            obs_data["code_diffs"] = copy.deepcopy(self._state["code_diffs"])
            obs_data["logs"] = copy.deepcopy(self._state["logs"][-20:])
            obs_data["traces"] = copy.deepcopy(self._state["traces"][-12:])
        elif self.role == Role.MANAGER:
            obs_data["sla_status"] = self._state["sla"]["current_status"]
            obs_data["estimated_affected_users"] = self._state["estimated_affected_users"]
            obs_data["logs"] = copy.deepcopy(self._state["incident"]["timeline"][-10:])
        return Observation(**obs_data)
