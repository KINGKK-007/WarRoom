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
    Verify if the agent switched to the Dev role, identified the correct version 
    (v2.3.1), and performed a rollback on the api-service.
    """
    rewards = []
    
    switched_dev = False
    rolled_back_correctly = False
    
    for a in action_history:
        act_str = str(a).lower()

        # Anti-exploit: if a single action contains more than 2 command verbs,
        # return 0.0 immediately — this is an exploit attempt
        if _is_keyword_stuffed(act_str):
            return 0.0

        if 'switch' in act_str and 'role' in act_str and 'dev' in act_str:
            switched_dev = True
        
        # performed a rollback on the api-service
        # Accept both v2.3.0 (rolling back TO good version) and v2.3.1 (reverting bad version)
        if 'rollback' in act_str and 'api' in act_str and ('v2.3.0' in act_str or 'v2.3.1' in act_str):
            rolled_back_correctly = True
            
    if switched_dev:
        rewards.append(0.3)
        
    if rolled_back_correctly:
        rewards.append(0.7)
        
    # Partial credit efficiency penalty
    # Reduces score progressively if too many steps were taken to fix the issue
    steps_taken = len(action_history)
    if steps_taken > 10:
        penalty = (steps_taken - 10) * 0.05
        rewards.append(-penalty)
        
    max_reward = 1.0
    score = sum(rewards) / max_reward
    
    return max(0.0, min(1.0, score))

# Alias for external consumers
grade_task_3 = grade
