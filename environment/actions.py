import re
from typing import Any, Dict, List


ACTION_CATALOG: List[str] = [
    "restart_service",
    "rollback_deploy",
    "switch_role",
    "scale",
    "inspect",
    "query_metrics",
    "query_deploy",
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
    "escalate",
    "notify",
    "update_status_page",
    "attach_runbook",
    "verify_sla",
    "run_rca",
    "generate_postmortem",
]


class CommandParser:
    def __init__(self):
        self.patterns = [
            (r"(?i)^restart\s+service\s+(?P<target>.+)$", "restart_service"),
            (r"(?i)^restart\s+(?P<target>.+)$", "restart_service"),
            (r"(?i)^rollback\s+deploy\s+(?P<target>\S+)\s+(?P<version>\S+)$", "rollback_deploy"),
            (r"(?i)^rollback\s+(?P<target>\S+)\s+(?P<version>\S+)$", "rollback_deploy"),
            (r"(?i)^switch\s+role\s+(?P<target>.+)$", "switch_role"),
            (r"(?i)^scale\s+(?P<target>\S+)(?:\s+(?P<count>\d+))?$", "scale"),
            (r"(?i)^inspect\s+(?P<target>.+)$", "inspect"),
            (r"(?i)^query\s+metrics$", "query_metrics"),
            (r"(?i)^check\s+metrics$", "query_metrics"),
            (r"(?i)^query\s+deploy(?:\s+history)?$", "query_deploy"),
            (r"(?i)^query\s+logs$", "query_logs"),
            (r"(?i)^view\s+logs$", "query_logs"),
            (r"(?i)^query\s+traces$", "query_traces"),
            (r"(?i)^query\s+topology$", "query_topology"),
            (r"(?i)^show\s+topology$", "query_topology"),
            (r"(?i)^run\s+health\s+check(?:\s+(?P<target>.+))?$", "run_health_check"),
            (r"(?i)^health\s+check(?:\s+(?P<target>.+))?$", "run_health_check"),
            (r"(?i)^failover\s+zone\s+(?P<target>.+)$", "failover_zone"),
            (r"(?i)^drain\s+zone\s+(?P<target>.+)$", "drain_zone"),
            (r"(?i)^restore\s+zone\s+(?P<target>.+)$", "restore_zone"),
            (r"(?i)^rebalance\s+traffic\s+(?P<target>.+)$", "rebalance_traffic"),
            (r"(?i)^clear\s+queue\s+(?P<target>.+)$", "clear_queue"),
            (r"(?i)^tune\s+autoscaling\s+(?P<target>.+)$", "tune_autoscaling"),
            (r"(?i)^throttle\s+service\s+(?P<target>.+)$", "throttle_service"),
            (r"(?i)^isolate\s+service\s+(?P<target>.+)$", "isolate_service"),
            (r"(?i)^acknowledge\s+alert(?:\s+(?P<target>.+))?$", "acknowledge_alert"),
            (r"(?i)^escalate\s+(?P<target>.+)$", "escalate"),
            (r"(?i)^notify\s+(?P<target>.+)$", "notify"),
            (r"(?i)^update\s+status\s+page(?:\s+(?P<target>.+))?$", "update_status_page"),
            (r"(?i)^attach\s+runbook(?:\s+(?P<target>.+))?$", "attach_runbook"),
            (r"(?i)^verify\s+sla$", "verify_sla"),
            (r"(?i)^run\s+rca$", "run_rca"),
            (r"(?i)^generate\s+postmortem$", "generate_postmortem"),
        ]
        self.compiled_patterns = [(re.compile(pattern), action) for pattern, action in self.patterns]

    def parse(self, command: str) -> Dict[str, Any]:
        command = command.strip()
        for pattern, action_name in self.compiled_patterns:
            match = pattern.match(command)
            if match:
                result = {"action": action_name}
                result.update({k: v for k, v in match.groupdict().items() if v is not None})
                return result
        return {"action": "unknown"}
