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
    Checks if restart service db was called and if final error_rate is < 0.05.
    """
    rewards = []
    
    # 1. Check if restart service db was called
    db_restarted = False
    other_restarted = False
    for a in action_history:
        act_str = str(a).lower()

        # Anti-exploit: skip keyword-stuffed actions
        if _is_keyword_stuffed(act_str):
            continue

        if 'restart' in act_str:
            if 'db' in act_str or 'database' in act_str:
                db_restarted = True
            else:
                other_restarted = True
            
    if db_restarted:
        rewards.append(0.5)
    elif other_restarted:
        rewards.append(0.3)

        
    # 2. Check if the final error_rate is < 0.05
    error_rate = final_state.get('metrics', {}).get('error_rate', 1.0)
    if error_rate < 0.05:
        rewards.append(0.5)
        
    # Partial credit for steps taken (efficiency penalty)
    # An agent taking too many steps receives a small penalty
    steps_taken = len(action_history)
    if steps_taken > 5:
        penalty = (steps_taken - 5) * 0.05
        rewards.append(-penalty)
        
    max_reward = 1.0
    score = sum(rewards) / max_reward
    
    return max(0.0, min(1.0, score))

# Aliases for external consumers
grade_task_1 = grade
_is_stuffed = _is_keyword_stuffed
