import re
from typing import Dict, Any

class CommandParser:
    def __init__(self):
        # Case insensitive regex patterns for matching agent queries
        self.patterns = [
            (r'(?i)^restart\s+service\s+(?P<target>.+)$', "restart_service"),
            (r'(?i)^rollback\s+deploy\s+(?P<target>\S+)\s+(?P<version>\S+)$', "rollback_deploy"),
            (r'(?i)^rollback\s+(?P<target>\S+)\s+(?P<version>\S+)$', "rollback_deploy"),
            (r'(?i)^switch\s+role\s+(?P<target>.+)$', "switch_role"),
            (r'(?i)^scale\s+(?P<target>.+)$', "scale"),
            (r'(?i)^inspect\s+(?P<target>.+)$', "inspect"),
            (r'(?i)^query\s+metrics$', "query_metrics"),
            (r'(?i)^query\s+deploy\s+history$', "query_deploy"),
            (r'(?i)^query\s+deploy$', "query_deploy"),
            (r'(?i)^query\s+logs$', "query_logs"),
            (r'(?i)^escalate\s+(?P<target>.+)$', "escalate"),
            (r'(?i)^notify\s+(?P<target>.+)$', "notify")
        ]
        self.compiled_patterns = [(re.compile(p), action) for p, action in self.patterns]

    def parse(self, command: str) -> Dict[str, Any]:
        """Parses a raw string command into a structured action dictionary."""
        command = command.strip()
        for pattern, action_name in self.compiled_patterns:
            match = pattern.match(command)
            if match:
                result = {"action": action_name}
                if match.groupdict():
                    result.update(match.groupdict())
                return result
        
        # Unrecognizable command
        return {"action": "unknown"}
