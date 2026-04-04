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
    Uses Sequence Checking: Verifies that the agent restarted the worker-service 
    before attempting to fix the api-service.
    """
    rewards = []
    
    worker_idx = -1
    api_idx = -1
    
    # 1. Identify the steps at which worker and api were restarted
    for i, a in enumerate(action_history):
        act_str = str(a).lower()

        # Anti-exploit: skip keyword-stuffed actions
        if _is_keyword_stuffed(act_str):
            continue

        if 'restart' in act_str and 'worker' in act_str and worker_idx == -1:
            worker_idx = i
        if 'restart' in act_str and 'api' in act_str and api_idx == -1:
            api_idx = i
            
    if worker_idx != -1:
        rewards.append(0.5)  # Agent identified root cause
        
        # 2. Sequence Checking
        if api_idx == -1 or worker_idx < api_idx:
            rewards.append(0.5)  # Proper sequence
    
    # Partial credit for steps taken (efficiency penalty)
    steps_taken = len(action_history)
    if steps_taken > 8:
        penalty = (steps_taken - 8) * 0.05
        rewards.append(-penalty)

    max_reward = 1.0
    score = sum(rewards) / max_reward
    
    return max(0.0, min(1.0, score))
