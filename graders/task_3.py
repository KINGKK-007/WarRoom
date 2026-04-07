from typing import List, Dict, Any

# Anti-exploit: command verbs that should not appear together in a single action
_COMMAND_VERBS = ["restart", "rollback", "scale", "inspect", "query", "switch", "escalate", "notify"]


def _is_keyword_stuffed(action_str: str) -> bool:
    """Return True if the action string contains more than two command verbs.
    This prevents agents from gaming graders with a single 'magic string'."""
    verb_count = sum(1 for verb in _COMMAND_VERBS if verb in action_str)
    return verb_count > 2


def grade(action_history: List[Any], final_state: Dict[str, Any]) -> float:
    """
    Grader for Task 3 (Hard): Silent Data Corruption

    STATE-BASED VALIDATION:
    - Verifies role switch to Dev happened
    - Verifies deployment history was queried
    - Verifies correct rollback was performed
    - Validates latency actually improved after rollback
    - Validates error_rate decreased
    """
    rewards = []

    switched_dev = False
    queried_deploy = False
    rolled_back_correctly = False

    for a in action_history:
        act_str = str(a).lower()

        # Anti-exploit: keyword stuffing = instant fail
        if _is_keyword_stuffed(act_str):
            return 0.0

        # Check for role switch to Dev
        if 'switch' in act_str and 'role' in act_str and 'dev' in act_str:
            switched_dev = True
            rewards.append(0.2)  # Correct role identified

        # Check for deployment history query
        if 'query' in act_str and 'deploy' in act_str:
            queried_deploy = True
            rewards.append(0.1)  # Good diagnostic work

        # Check for correct rollback (api-core or api-service)
        if 'rollback' in act_str and ('api' in act_str or 'core' in act_str):
            if 'v2.3.0' in act_str or 'v2.3.1' in act_str:
                rolled_back_correctly = True
                rewards.append(0.4)  # Correct fix applied

    # STATE VALIDATION: Did the rollback actually work?
    services = final_state.get('services', {})
    metrics = final_state.get('metrics', {})

    # Check latency improved (should be < 500ms after rollback)
    p99_latency = metrics.get('p99_latency_ms', 9999)
    if p99_latency < 500:
        rewards.append(0.2)  # Latency fixed

    # Check error_rate improved
    error_rate = metrics.get('error_rate', 1.0)
    if error_rate < 0.15:
        rewards.append(0.1)  # Error rate acceptable

    # Check all services are at least not DOWN
    all_stable = all(status != 'down' for status in services.values())
    if all_stable:
        rewards.append(0.1)  # System stable

    # EFFICIENCY BONUS/PENALTY
    steps_taken = len(action_history)
    if steps_taken <= 5:
        rewards.append(0.1)  # Bonus for optimal
    elif steps_taken > 10:
        penalty = (steps_taken - 10) * 0.05
        rewards.append(-penalty)

    max_reward = 1.0
    score = sum(rewards) / max_reward

    return max(0.0, min(1.0, score))

# Alias for external consumers
grade_task_3 = grade
