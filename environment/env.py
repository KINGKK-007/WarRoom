import copy
from typing import Any, Dict, Tuple
from .models import Role, ServiceState, Scenario, Observation, Action, Reward, Alert
from .actions import CommandParser
from .roles import RoleManager
from .scenarios import SCENARIOS

try:
    from openenv import BaseEnvironment
except ImportError:
    class BaseEnvironment:
        pass

class DevOpsWarRoomEnv(BaseEnvironment):
    def __init__(self, role: Role = Role.SRE):
        self.role_manager = RoleManager(role)
        self.role = self.role_manager.current_role
        self.parser = CommandParser()
        self._state = {}
        self.history = {}
        self.pending_events = []
        self.tick_count = 0
        self.step_count = 0
        self.task_id = Scenario.EASY
        self.reset(self.task_id)

    @property
    def state(self) -> dict:
        """Return the current raw environment state (property to avoid attribute/method shadowing)."""
        return self._state

    def reset(self, task_id: Scenario = Scenario.EASY) -> Observation:
        self.step_count = 0
        self.tick_count = 0
        self.task_id = task_id
        self.pending_events = []
        
        # Reset role to SRE to prevent cross-episode state leak
        self.role_manager = RoleManager(Role.SRE)
        self.role = self.role_manager.current_role
        
        # Base state from scenarios
        self._state = copy.deepcopy(SCENARIOS[task_id])
        
        # internal tracking for cascading failures
        self.history = {
            'database_down_tick': None,
            'worker_degraded_tick': None
        }

        return self._get_observation()

    def step(self, action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        # Accept both Action model and raw dict
        if isinstance(action, dict):
            action = Action(**action)
        self.step_count += 1
        # Handle parsed matching
        if action.action_type in ["command", "raw_command"]:
            raw_command = action.target or (action.params.get("command") if action.params else "")
            parsed = self.parser.parse(str(raw_command))
        else:
            parsed = self.parser.parse(action.action_type)
        
        if parsed.get("action") and parsed.get("action") != "unknown":
            action_name = parsed.get("action")
            target = parsed.get("target")
            version = parsed.get("version")
        else:
            action_name = action.action_type
            target = action.target
            version = action.params.get("version") if action.params else None

        # Noise penalty for unrecognizable command
        if action_name == "unknown":
            self.tick()
            reward = Reward(value=-0.05, reason="unrecognizable command noise", done=self._is_done())
            return self._get_observation(), reward, self._is_done(), {"error": "unrecognizable command noise"}

        # Role Switch Penalty (1-tick delay)
        if action_name == "switch_role":
            self.tick()
            if target:
                self.role_manager.issue_switch_role(target)
                self.role = self.role_manager.current_role
            reward_reason = f"Role switched to {self.role.value}"
            reward = Reward(value=0.0, reason=reward_reason, done=self._is_done())
            return self._get_observation(), reward, self._is_done(), {"info": reward_reason}

        # Check authorization (SRE/Dev/Manager matrix)
        if not self.role_manager.check_action_allowed(action_name):
            # Unauthorized action: return -0.15 penalty without executing state change (nor ticking)
            reward = Reward(value=-0.15, reason="Unauthorized action for current role", done=self._is_done())
            return self._get_observation(), reward, self._is_done(), {"error": "Unauthorized action for current role"}

        # --- Informational actions: return state info with 0.0 reward, no dense penalty ---
        if action_name == "inspect":
            self.tick()
            info_data = {}
            if target and target in self._state['services']:
                svc_status = self._state['services'][target]
                info_data = {
                    "service": target,
                    "status": svc_status.value if isinstance(svc_status, ServiceState) else str(svc_status),
                    "error_rate": self._state['metrics']['error_rate'],
                }
            else:
                info_data = {"error": f"Unknown service: {target}"}
            reward = Reward(value=0.0, reason=f"Inspected {target}.", done=self._is_done())
            return self._get_observation(), reward, self._is_done(), info_data

        if action_name == "query_metrics":
            self.tick()
            reward = Reward(value=0.0, reason="Queried metrics.", done=self._is_done())
            return self._get_observation(), reward, self._is_done(), {"metrics": copy.deepcopy(self._state['metrics'])}

        if action_name == "query_deploy":
            self.tick()
            reward = Reward(value=0.0, reason="Queried deployment history.", done=self._is_done())
            return self._get_observation(), reward, self._is_done(), {"deploy_history": copy.deepcopy(self._state.get('deploy_history', []))}

        if action_name == "query_logs":
            self.tick()
            reward = Reward(value=0.0, reason="Queried logs.", done=self._is_done())
            return self._get_observation(), reward, self._is_done(), {"logs": copy.deepcopy(self._state.get('logs', [])[-30:])}

        if action_name == "scale":
            self.tick()
            reward = Reward(value=0.0, reason=f"Scaled {target}.", done=self._is_done())
            return self._get_observation(), reward, self._is_done(), {"info": f"Scaled {target}"}

        if action_name == "escalate":
            self.tick()
            reward = Reward(value=0.0, reason=f"Escalated {target}.", done=self._is_done())
            return self._get_observation(), reward, self._is_done(), {"info": f"Escalated {target}"}

        if action_name == "notify":
            self.tick()
            reward = Reward(value=0.0, reason=f"Notified {target}.", done=self._is_done())
            return self._get_observation(), reward, self._is_done(), {"info": f"Notified {target}"}

        # Execute authorized active state changes
        reward_value = 0.0
        reason_str = "Action processed."
        
        if action_name == "restart_service":
            if target and target in self._state['services']:
                if self._state['services'][target] in [ServiceState.degraded, ServiceState.down]:
                    self._state['services'][target] = ServiceState.healthy
                    reward_value += 0.40 # correct_diagnosis
                    reason_str = "Correct diagnosis and restart."
                else:
                    reward_value -= 0.20 # wrong_service_restart
                    reason_str = "Wrong service restart."
            else:
                reward_value -= 0.20 # wrong_service_restart (invalid target)
                reason_str = "Wrong service restart (invalid target)."
        
        elif action_name == "rollback_deploy":
            if target and target in self._state['services'] and version:
                if self._state['services'][target] in [ServiceState.degraded, ServiceState.down]:
                    self._state['services'][target] = ServiceState.healthy
                    reward_value += 0.40 # correct_diagnosis
                    reason_str = "Correct diagnosis and rollback."
                else:
                    reward_value -= 0.25 # wrong_rollback
                    reason_str = "Wrong rollback."
            else:
                reward_value -= 0.25 # wrong_rollback (invalid target or missing version)
                reason_str = "Wrong rollback (invalid target or missing version)."

        self.tick()
        
        # Dense signal: only apply penalties for genuinely wrong actions (strictly negative),
        # NOT for neutral informational actions (== 0.0)
        if reward_value < 0.0:
            has_critical = any(isinstance(a, Alert) and a.severity == "CRITICAL" for a in self._state['alerts'])
            if has_critical:
                reward_value -= 0.10
                if reason_str == "Action processed.":
                    reason_str = "Ignored critical alert."
            else:
                reward_value -= 0.15 # made_metrics_worse
                if reason_str == "Action processed.":
                    reason_str = "Made metrics worse."

        reward = Reward(value=reward_value, reason=reason_str, done=self._is_done())
        return self._get_observation(), reward, self._is_done(), {}

    def _is_done(self) -> bool:
        return (
            self.step_count >= 15
            or self._state['metrics'].get('error_rate', 0) >= 1.0
        )

    def tick(self):
        self.tick_count += 1
        
        # Base error rate increment per step (2%)
        self._state['metrics']['error_rate'] += 0.02
        
        # Recovery mechanic: when a service is UP, decrease error_rate toward baseline
        all_up = all(
            s == ServiceState.healthy for s in self._state['services'].values()
        )
        if all_up and self._state['metrics']['error_rate'] > 0.0:
            self._state['metrics']['error_rate'] = max(
                0.0,
                self._state['metrics']['error_rate'] - 0.05
            )
        
        # Trigger pending events
        events_to_keep = []
        for event in self.pending_events:
            if self.tick_count == event.get('tick'):
                event_type = event.get('event')
                if event_type == 'api_cascade_degraded':
                    if self._state['services']['api'] == ServiceState.healthy and self._state['services']['database'] == ServiceState.down:
                        self._state['services']['api'] = ServiceState.degraded
                        cascade_alert = Alert(
                            severity="CRITICAL",
                            message="[CASCADE] API degraded due to Database being DOWN for 2 ticks.",
                            fired_at=self.tick_count
                        )
                        self._state['alerts'].append(cascade_alert)
                elif event_type == 'worker_oom':
                    if self._state['services']['worker'] == ServiceState.degraded:
                        self._state['services']['worker'] = ServiceState.down
                        self._state['services']['auth'] = ServiceState.degraded
                        cascade_alert = Alert(
                            severity="CRITICAL",
                            message="[CASCADE] Worker OOM triggered Auth degradation.",
                            fired_at=self.tick_count
                        )
                        self._state['alerts'].append(cascade_alert)
            else:
                events_to_keep.append(event)
        self.pending_events = events_to_keep

        # Cascading failure logic
        if self._state['services']['database'] == ServiceState.down:
            if self.history['database_down_tick'] is None:
                self.history['database_down_tick'] = self.tick_count
                
                # Queue cascading failure
                self.pending_events.append({
                    'tick': self.tick_count + 2,
                    'event': 'api_cascade_degraded'
                })
        else:
            self.history['database_down_tick'] = None

        if self._state['services']['worker'] == ServiceState.degraded:
            if self.history['worker_degraded_tick'] is None:
                self.history['worker_degraded_tick'] = self.tick_count
                
                # Queue worker_oom failure (cascade to auth)
                self.pending_events.append({
                    'tick': self.tick_count + 2,
                    'event': 'worker_oom'
                })
        else:
            self.history['worker_degraded_tick'] = None

    def _get_observation(self) -> Observation:
        # Base observation parameters
        obs_data = {
            'tick': self.tick_count,
            'current_role': self.role.value,
            'services': copy.deepcopy(self._state['services']),
            'metrics': copy.deepcopy(self._state['metrics']),
            'alerts': copy.deepcopy(self._state['alerts']),
            'steps_remaining': max(0, 15 - self.step_count),
        }
        
        # Strict role-based payload filtering
        if self.role == Role.SRE:
            # SRE sees Metrics + Last 10 log lines
            obs_data['logs'] = self._state.get('logs', [])[-10:]
        elif self.role == Role.DEV:
            # Dev sees Deployment history + Last 30 logs + Code diffs
            obs_data.pop('metrics', None)
            obs_data['deployment_history'] = self._state.get('deploy_history', [])
            obs_data['code_diffs'] = self._state.get('code_diffs', [])
            obs_data['logs'] = self._state.get('logs', [])[-30:]
        elif self.role == Role.MANAGER:
            # Manager sees SLA status + Estimated affected users
            obs_data.pop('metrics', None)
            obs_data['sla_status'] = self._state.get('sla_status', "Unknown")
            # For demonstration, estimated affected users is derived from error_rate
            obs_data['estimated_affected_users'] = int(self._state['metrics']['error_rate'] * 10000)

        return Observation(**obs_data)
