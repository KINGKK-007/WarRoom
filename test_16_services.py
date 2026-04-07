"""
Test script for 16-service expanded environment
Tests all 3 tasks without requiring LLM API
"""
from environment.env import DevOpsWarRoomEnv
from environment.models import Scenario
from graders.task_1 import grade as grade_task_1
from graders.task_2 import grade as grade_task_2
from graders.task_3 import grade as grade_task_3

def test_task_1():
    """Test Task 1: Postgres outage"""
    print("=== TASK 1: PostgreSQL Outage ===")
    env = DevOpsWarRoomEnv()
    env.reset(Scenario.Easy)

    # Execute optimal solution
    actions = []

    # Step 1: Restart postgres
    obs, reward, done, info = env.step({'action_type': 'raw_command', 'params': {'command': 'restart service postgres'}})
    actions.append('restart service postgres')
    print(f"Step 1: restart service postgres | Reward: {reward.value}")

    # Step 2: Verify
    obs, reward, done, info = env.step({'action_type': 'raw_command', 'params': {'command': 'query metrics'}})
    actions.append('query metrics')
    print(f"Step 2: query metrics | Reward: {reward.value}")

    # Get final state and grade
    final_state = {'services': dict(env.state['services']), 'metrics': env.state['metrics']}
    score = grade_task_1(actions, final_state)

    print(f"Services: postgres={env.state['services']['postgres']}, error_rate={env.state['metrics']['error_rate']:.3f}")
    print(f"Grader Score: {score:.2f} | Target: >=0.65 | {'PASS' if score >= 0.65 else 'FAIL'}")
    print()
    return score

def test_task_2():
    """Test Task 2: Worker cascading failure"""
    print("=== TASK 2: Worker Cascading Failure ===")
    env = DevOpsWarRoomEnv()
    env.reset(Scenario.Medium)

    actions = []

    # Step 1: Restart worker
    obs, reward, done, info = env.step({'action_type': 'raw_command', 'params': {'command': 'restart service worker'}})
    actions.append('restart service worker')
    print(f"Step 1: restart service worker | Reward: {reward.value}")

    # Step 2: Verify
    obs, reward, done, info = env.step({'action_type': 'raw_command', 'params': {'command': 'query metrics'}})
    actions.append('query metrics')
    print(f"Step 2: query metrics | Reward: {reward.value}")

    final_state = {'services': dict(env.state['services']), 'metrics': env.state['metrics']}
    score = grade_task_2(actions, final_state)

    print(f"Services: worker={env.state['services']['worker']}, error_rate={env.state['metrics']['error_rate']:.3f}")
    print(f"Grader Score: {score:.2f} | Target: >=0.40 | {'PASS' if score >= 0.40 else 'FAIL'}")
    print()
    return score

def test_task_3():
    """Test Task 3: Silent data corruption"""
    print("=== TASK 3: Silent Data Corruption (Bad Deployment) ===")
    env = DevOpsWarRoomEnv()
    env.reset(Scenario.Hard)

    actions = []

    # Step 1: Switch to Dev role
    obs, reward, done, info = env.step({'action_type': 'raw_command', 'params': {'command': 'switch role dev'}})
    actions.append('switch role dev')
    print(f"Step 1: switch role dev | Reward: {reward.value}")

    # Step 2: Query deploy history
    obs, reward, done, info = env.step({'action_type': 'raw_command', 'params': {'command': 'query deploy history'}})
    actions.append('query deploy history')
    print(f"Step 2: query deploy history | Reward: {reward.value}")

    # Step 3: Rollback api-core
    obs, reward, done, info = env.step({'action_type': 'raw_command', 'params': {'command': 'rollback api-core v2.3.0'}})
    actions.append('rollback api-core v2.3.0')
    print(f"Step 3: rollback api-core v2.3.0 | Reward: {reward.value}")

    # Step 4: Verify
    obs, reward, done, info = env.step({'action_type': 'raw_command', 'params': {'command': 'query metrics'}})
    actions.append('query metrics')
    print(f"Step 4: query metrics | Reward: {reward.value}")

    final_state = {'services': dict(env.state['services']), 'metrics': env.state['metrics']}
    score = grade_task_3(actions, final_state)

    print(f"Services: api-core={env.state['services']['api-core']}")
    print(f"Grader Score: {score:.2f} | Target: >=0.15 | {'PASS' if score >= 0.15 else 'FAIL'}")
    print()
    return score

def test_dependency_cascades():
    """Test dependency-based cascading failures"""
    print("=== DEPENDENCY CASCADE TEST ===")
    env = DevOpsWarRoomEnv()
    env.reset(Scenario.Easy)

    print(f"Initial: postgres={env.state['services']['postgres']}, api-core={env.state['services']['api-core']}, user-service={env.state['services']['user-service']}")

    # Tick a few times to let cascades propagate
    for i in range(3):
        env.tick()
        print(f"Tick {i+1}: postgres={env.state['services']['postgres']}, api-core={env.state['services']['api-core']}, user-service={env.state['services']['user-service']}")

    # api-core and user-service should degrade because they depend on postgres
    api_core_degraded = env.state['services']['api-core'] == 'degraded'
    user_service_degraded = env.state['services']['user-service'] == 'degraded'

    print(f"Cascade working: api-core degraded={api_core_degraded}, user-service degraded={user_service_degraded}")
    print()

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING 16-SERVICE EXPANDED ENVIRONMENT")
    print("=" * 60)
    print()

    # Test dependency system
    test_dependency_cascades()

    # Test all 3 tasks
    score1 = test_task_1()
    score2 = test_task_2()
    score3 = test_task_3()

    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Task 1 Score: {score1:.2f} | {'PASS' if score1 >= 0.65 else 'FAIL'}")
    print(f"Task 2 Score: {score2:.2f} | {'PASS' if score2 >= 0.40 else 'FAIL'}")
    print(f"Task 3 Score: {score3:.2f} | {'PASS' if score3 >= 0.15 else 'FAIL'}")
    print()

    all_pass = score1 >= 0.65 and score2 >= 0.40 and score3 >= 0.15
    print(f"OVERALL: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
