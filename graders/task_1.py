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
    Grader for Task 1 (Easy): Single Service Outage

    STATE-BASED VALIDATION (not just keywords):
    - Verifies database service actually became healthy
    - Verifies error_rate actually dropped below threshold
    - Penalizes unnecessary restarts of healthy services
    - Rewards efficiency (fewer steps)
    """
    rewards = []

    # 1. ACTION VALIDATION: Check if correct service was restarted
    db_restarted = False
    unnecessary_restarts = 0

    for a in action_history:
        act_str = str(a).lower()

        # Anti-exploit: skip keyword-stuffed actions
        if _is_keyword_stuffed(act_str):
            continue

        if 'restart' in act_str:
            if 'db' in act_str or 'database' in act_str or 'postgres' in act_str:
                db_restarted = True
                rewards.append(0.4)  # Correct diagnosis
            else:
                # Restarted wrong service (api-core, worker, auth, etc.)
                unnecessary_restarts += 1
                rewards.append(-0.15)  # Penalty for wrong action

    # 2. STATE VALIDATION: Did the fix actually work?
    services = final_state.get('services', {})
    metrics = final_state.get('metrics', {})
    error_rate = metrics.get('error_rate', 1.0)

    # Check postgres is now healthy (backwards compatible with 'database')
    db_status = services.get('postgres', services.get('database', 'unknown'))
    if db_status == 'healthy':
        rewards.append(0.3)  # Database recovered

    # Check error_rate improved
    if error_rate < 0.05:
        rewards.append(0.3)  # System recovered

    # 3. EFFICIENCY BONUS/PENALTY
    steps_taken = len(action_history)
    if steps_taken <= 3:
        rewards.append(0.1)  # Bonus for optimal solution
    elif steps_taken > 5:
        penalty = (steps_taken - 5) * 0.05
        rewards.append(-penalty)

    max_reward = 1.0
    score = sum(rewards) / max_reward

    return max(0.0, min(1.0, score))

# Aliases for external consumers
grade_task_1 = grade
_is_stuffed = _is_keyword_stuffed
