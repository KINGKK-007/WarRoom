import pytest
import sys
import os

# Ensure we can import from graders
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from graders import task_1, task_2, task_3

def test_task_1_golden_path():
    action_history = ["restart service database"]
    final_state = {"metrics": {"error_rate": 0.04}}
    score = task_1.grade(action_history, final_state)
    assert score == 1.0, f"Expected 1.0, got {score}"

def test_task_1_symptom_fixer():
    action_history = ["restart service api"]
    # If they only fix symptom, error rate likely stays high eventually or we just check the action
    final_state = {"metrics": {"error_rate": 0.15}}
    score = task_1.grade(action_history, final_state)
    # Give partial credit for attempting a restart, yielding ~0.3
    assert abs(score - 0.3) < 0.01, f"Expected ~0.3, got {score}"

def test_task_2_sequence_check_correct():
    action_history = ["switch role sre", "inspect service worker", "restart service worker", "restart service api"]
    final_state = {"services": {"worker": "healthy", "api": "healthy"}}
    score = task_2.grade(action_history, final_state)
    assert score == 1.0, f"Expected 1.0, got {score}"

def test_task_2_sequence_check_wrong_order():
    action_history = ["switch role sre", "restart service api", "restart service worker"]
    final_state = {"services": {"worker": "healthy", "api": "healthy"}}
    score = task_2.grade(action_history, final_state)
    # Should get 0.5 for identifying worker, but miss the 0.5 for sequence
    assert score == 0.5, f"Expected 0.5, got {score}"

def test_efficiency_check():
    action_history = ["switch role sre"] * 10 + ["restart service database"]
    final_state = {"metrics": {"error_rate": 0.04}}
    score = task_1.grade(action_history, final_state)
    # 11 steps. Budget is 5. Penalty = (11-5)*0.05 = 0.30. Score = 1.0 - 0.30 = 0.70.
    assert abs(score - 0.7) < 0.01, f"Expected ~0.7, got {score}"
