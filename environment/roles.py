from .actions import ACTION_CATALOG
from .models import Role


class RoleManager:
    def __init__(self, initial_role: Role = Role.SRE):
        self.current_role = initial_role
        self.permissions = {
            Role.SRE: {
                "restart_service",
                "scale",
                "inspect",
                "query_metrics",
                "query_logs",
                "query_traces",
                "query_topology",
                "run_health_check",
                "failover_zone",
                "drain_zone",
                "restore_zone",
                "rebalance_traffic",
                "clear_queue",
                "tune_autoscaling",
                "throttle_service",
                "isolate_service",
                "acknowledge_alert",
                "switch_role",
                "verify_sla",
                "run_rca",
            },
            Role.DEV: {
                "rollback_deploy",
                "query_deploy",
                "query_logs",
                "query_traces",
                "inspect",
                "isolate_service",
                "attach_runbook",
                "switch_role",
                "run_rca",
                "generate_postmortem",
            },
            Role.MANAGER: {
                "escalate",
                "notify",
                "update_status_page",
                "verify_sla",
                "run_rca",
                "generate_postmortem",
                "switch_role",
                "attach_runbook",
            },
        }

    def check_action_allowed(self, action_name: str) -> bool:
        if action_name == "unknown":
            return False
        return action_name in self.permissions.get(self.current_role, set())

    def issue_switch_role(self, new_role_str: str) -> bool:
        for role in Role:
            if role.value.lower() == new_role_str.lower():
                self.current_role = role
                return True
        return False

    def available_actions(self):
        allowed = self.permissions.get(self.current_role, set())
        return [action for action in ACTION_CATALOG if action in allowed]
