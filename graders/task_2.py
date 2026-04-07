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
    Grader for Task 2 (Medium): Cascading Failure

    STATE-BASED VALIDATION:
    - Verifies worker was actually fixed (became healthy)
    - Verifies cascade was prevented (api/auth didn't degrade)
    - Checks proper sequence (worker before api if both touched)
    - Validates error_rate improved
    """
    rewards = []

    worker_idx = -1
    api_idx = -1
    unnecessary_restarts = 0

    # 1. ACTION VALIDATION: Identify restarts
    for i, a in enumerate(action_history):
        act_str = str(a).lower()

        # Anti-exploit: skip keyword-stuffed actions
        if _is_keyword_stuffed(act_str):
            continue

        if 'restart' in act_str:
            if 'worker' in act_str and worker_idx == -1:
                worker_idx = i
                rewards.append(0.4)  # Root cause identified
            elif 'api' in act_str and api_idx == -1:
                api_idx = i
            elif 'database' in act_str or 'db' in act_str:
                # Restarted wrong service
                unnecessary_restarts += 1
                rewards.append(-0.1)

    # 2. SEQUENCE VALIDATION: Worker before API (if both restarted)
    if worker_idx != -1 and (api_idx == -1 or worker_idx < api_idx):
        rewards.append(0.3)  # Correct sequence

    # 3. STATE VALIDATION: Did the fix work?
    services = final_state.get('services', {})
    metrics = final_state.get('metrics', {})

    # Check worker is now healthy
    worker_status = services.get('worker', 'unknown')
    if worker_status == 'healthy':
        rewards.append(0.2)  # Worker recovered

    # Check cascade was prevented (api and auth still healthy)
    api_status = services.get('api', 'unknown')
    auth_status = services.get('auth', 'unknown')
    if api_status == 'healthy' and auth_status == 'healthy':
        rewards.append(0.2)  # Cascade prevented

    # Check error_rate is acceptable
    error_rate = metrics.get('error_rate', 1.0)
    if error_rate < 0.1:
        rewards.append(0.1)  # System stable

    # 4. EFFICIENCY BONUS/PENALTY
    steps_taken = len(action_history)
    if steps_taken <= 4:
        rewards.append(0.1)  # Bonus for optimal
    elif steps_taken > 8:
        penalty = (steps_taken - 8) * 0.05
        rewards.append(-penalty)

    max_reward = 1.0
    score = sum(rewards) / max_reward

    return max(0.0, min(1.0, score))
