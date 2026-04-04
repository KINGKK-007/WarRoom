from .models import Role

class RoleManager:
    def __init__(self, initial_role: Role = Role.SRE):
        self.current_role = initial_role
        
        # Authorization Matrix defined by explicit constraints
        self.permissions = {
            Role.SRE: {"restart_service", "scale", "inspect", "query_metrics", "switch_role"},
            Role.DEV: {"rollback_deploy", "query_deploy", "query_logs", "switch_role"},
            Role.MANAGER: {"escalate", "notify", "switch_role"}
        }

    def check_action_allowed(self, action_name: str) -> bool:
        """Returns True if the current role is allowed to execute the action_name."""
        if action_name == "unknown":
            return False
            
        allowed_actions = self.permissions.get(self.current_role, set())
        return action_name in allowed_actions

    def issue_switch_role(self, new_role_str: str) -> bool:
        """Attempts to update current role. Returns True if successful."""
        for role in Role:
            if role.value.lower() == new_role_str.lower():
                self.current_role = role
                return True
        return False
