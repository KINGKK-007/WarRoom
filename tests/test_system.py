"""
tests/test_system.py — End-to-End System Validation & Spec Compliance

This test suite simulates full agent episodes for all three scenarios
(Easy, Medium, Hard) and validates:

1. Scenario Integrity          — Correct initial states from scenarios.py
2. Role-Switching Blindness    — SRE vs Manager observation filtering
3. Cascading Failure (2-Tick)  — Pending events queue determinism
4. Authorization Enforcement   — Role-action permission matrix
5. Pydantic Contract           — Reward model structure
6. Hard Task Solving           — Correct multi-step resolution path
"""

import copy
import pytest
from environment.models import (
    Role, ServiceState, Scenario, Observation, Action, Reward, Alert,
)
from environment.scenarios import SCENARIOS
from environment.env import DevOpsWarRoomEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw_action(command: str) -> Action:
    """Build a raw-command Action the way an LLM agent would submit it."""
    return Action(action_type="command", target=command)


def _make_typed_action(action_type: str, target: str = None, params: dict = None) -> Action:
    return Action(action_type=action_type, target=target, params=params)


# =========================================================================
# 1. Scenario Integrity
# =========================================================================

class TestScenarioIntegrity:
    """Verify that env.reset(task_id=...) loads the exact initial states
    defined in scenarios.py for EASY and MEDIUM."""

    def test_easy_initial_state(self):
        env = DevOpsWarRoomEnv()
        obs = env.reset(task_id=Scenario.EASY)

        expected = SCENARIOS[Scenario.EASY]

        # Service states must match exactly
        assert obs.services["database"] == ServiceState.down
        assert obs.services["api"] == ServiceState.healthy
        assert obs.services["frontend"] == ServiceState.healthy
        assert obs.services["worker"] == ServiceState.healthy
        assert obs.services["auth"] == ServiceState.healthy

        # Tick must be 0 after reset
        assert obs.tick == 0

        # Metrics error_rate should be 0.0 (SRE default role sees metrics)
        assert obs.metrics is not None
        assert obs.metrics["error_rate"] == 0.0

        # There should be exactly one CRITICAL alert
        assert len(obs.alerts) == 1
        assert obs.alerts[0].severity == "CRITICAL"
        assert "5432" in obs.alerts[0].message

    def test_medium_initial_state(self):
        env = DevOpsWarRoomEnv()
        obs = env.reset(task_id=Scenario.MEDIUM)

        # Service states
        assert obs.services["database"] == ServiceState.healthy
        assert obs.services["worker"] == ServiceState.degraded

        # Tick is 0
        assert obs.tick == 0

        # Metrics (SRE role) — error_rate should be 0.05
        assert obs.metrics is not None
        assert obs.metrics["error_rate"] == pytest.approx(0.05)

        # No alerts in MEDIUM scenario
        assert len(obs.alerts) == 0

    def test_hard_initial_state(self):
        env = DevOpsWarRoomEnv()
        obs = env.reset(task_id=Scenario.HARD)

        assert obs.services["database"] == ServiceState.degraded
        assert obs.services["api"] == ServiceState.healthy
        assert obs.tick == 0
        assert obs.metrics["error_rate"] == pytest.approx(0.10)
        # Expanded metrics must be present
        assert "cpu" in obs.metrics
        assert "memory" in obs.metrics
        assert "p99_latency_ms" in obs.metrics
        assert "requests_per_sec" in obs.metrics
        assert len(obs.alerts) == 0

    def test_reset_clears_pending_events(self):
        """After a full episode, reset must not leak events into the next."""
        env = DevOpsWarRoomEnv()
        env.reset(task_id=Scenario.EASY)

        # Run a few steps to build up pending events
        for _ in range(3):
            env.step(_make_raw_action("inspect database"))

        assert len(env.pending_events) > 0 or env.tick_count > 0

        # Now reset to MEDIUM — events list and tick must be clean
        obs = env.reset(task_id=Scenario.MEDIUM)
        assert env.pending_events == []
        assert env.tick_count == 0
        assert obs.tick == 0


# =========================================================================
# 2. Role-Switching "Blindness"
# =========================================================================

class TestRoleSwitchingBlindness:
    """SRE sees metrics but NOT sla_status.
    Manager sees sla_status but NOT raw metrics (like error_rate)."""

    def test_sre_sees_metrics_not_sla(self):
        env = DevOpsWarRoomEnv(role=Role.SRE)
        obs = env.reset(task_id=Scenario.EASY)

        # SRE must see metrics
        assert obs.metrics is not None
        assert "error_rate" in obs.metrics

        # SRE must NOT see sla_status
        assert obs.sla_status is None

    def test_manager_sees_sla_not_metrics(self):
        env = DevOpsWarRoomEnv(role=Role.SRE)
        env.reset(task_id=Scenario.EASY)

        # Switch to Manager
        obs, reward, done, info = env.step(_make_raw_action("switch role Manager"))

        # Manager must see sla_status
        assert obs.sla_status is not None

        # Manager must NOT see raw metrics
        assert obs.metrics is None

    def test_dev_sees_deployment_history(self):
        """Dev role observes deployment_history and code_diffs but not metrics."""
        env = DevOpsWarRoomEnv(role=Role.SRE)
        env.reset(task_id=Scenario.EASY)

        obs, _, _, _ = env.step(_make_raw_action("switch role Dev"))

        assert obs.metrics is None
        assert obs.deployment_history is not None
        assert len(obs.deployment_history) > 0
        assert obs.code_diffs is not None

    def test_sre_sees_logs(self):
        """SRE role receives the last 10 log lines."""
        env = DevOpsWarRoomEnv(role=Role.SRE)
        obs = env.reset(task_id=Scenario.EASY)
        assert obs.logs is not None
        assert len(obs.logs) <= 10


# =========================================================================
# 3. Cascading Failure — The 2-Tick Clock
# =========================================================================

class TestCascadingFailure:
    """In Task 1 (EASY), the DB is already DOWN.
    After exactly 2 noise ticks, the API must cascade to DEGRADED via
    the pending_events queue."""

    def test_api_cascades_after_two_ticks(self):
        env = DevOpsWarRoomEnv()
        obs = env.reset(task_id=Scenario.EASY)

        # Pre-conditions
        assert obs.services["database"] == ServiceState.down
        assert obs.services["api"] == ServiceState.healthy

        # Tick 1 — noise action
        obs1, _, _, _ = env.step(_make_raw_action("inspect database"))
        assert obs1.services["api"] == ServiceState.healthy, \
            "API should still be UP after 1 tick"

        # Tick 2 — noise action (but DB was detected DOWN on tick 1)
        # The cascade event was queued at tick_count+2 = 3 during the first
        # tick call. So we need a third tick for the event to fire.
        obs2, _, _, _ = env.step(_make_raw_action("inspect database"))

        # Tick 3 — this is when the cascade event fires (queued_tick = 1 + 2 = 3)
        obs3, _, _, _ = env.step(_make_raw_action("inspect database"))
        assert obs3.services["api"] == ServiceState.degraded, \
            "API should be DEGRADED after the cascade event fires"

        # Verify a CASCADE alert was appended
        cascade_alerts = [a for a in obs3.alerts if "CASCADE" in a.message]
        assert len(cascade_alerts) >= 1

    def test_cascade_event_does_not_leak_after_reset(self):
        """Pending events from one episode must not leak into the next."""
        env = DevOpsWarRoomEnv()
        env.reset(task_id=Scenario.EASY)

        # Trigger cascade setup
        env.step(_make_raw_action("inspect database"))  # tick 1

        # Reset mid-episode
        obs = env.reset(task_id=Scenario.MEDIUM)
        assert env.pending_events == []

        # Step twice in MEDIUM — API should remain UP (no lingering cascade)
        obs1, _, _, _ = env.step(_make_raw_action("inspect worker"))
        obs2, _, _, _ = env.step(_make_raw_action("inspect worker"))
        assert obs2.services["api"] == ServiceState.healthy


# =========================================================================
# 4. Authorization Enforcement
# =========================================================================

class TestAuthorizationEnforcement:
    """While in Manager role, attempt `restart service database`.
    Assert: reward is exactly -0.15 and DB status remains DOWN."""

    def test_manager_cannot_restart_service(self):
        env = DevOpsWarRoomEnv()
        env.reset(task_id=Scenario.EASY)

        # Switch to Manager first
        env.step(_make_raw_action("switch role Manager"))

        # Attempt unauthorized action
        obs, reward, done, info = env.step(
            _make_raw_action("restart service database")
        )

        # Reward must be exactly -0.15
        assert reward.value == pytest.approx(-0.15), \
            f"Expected -0.15, got {reward.value}"

        # DB must still be DOWN (action was not executed)
        assert obs.services["database"] == ServiceState.down

    def test_manager_cannot_rollback(self):
        env = DevOpsWarRoomEnv()
        env.reset(task_id=Scenario.EASY)
        env.step(_make_raw_action("switch role Manager"))

        _, reward, _, _ = env.step(
            _make_raw_action("rollback deploy database v1.0.0")
        )
        assert reward.value == pytest.approx(-0.15)

    def test_sre_cannot_rollback(self):
        """SRE is not authorized to rollback deployments."""
        env = DevOpsWarRoomEnv(role=Role.SRE)
        env.reset(task_id=Scenario.EASY)

        _, reward, _, _ = env.step(
            _make_raw_action("rollback deploy database v1.0.0")
        )
        assert reward.value == pytest.approx(-0.15)

    def test_dev_cannot_restart(self):
        """Dev is not authorized to restart services."""
        env = DevOpsWarRoomEnv(role=Role.SRE)
        env.reset(task_id=Scenario.EASY)
        env.step(_make_raw_action("switch role Dev"))

        _, reward, _, _ = env.step(
            _make_raw_action("restart service database")
        )
        assert reward.value == pytest.approx(-0.15)


# =========================================================================
# 5. Pydantic Contract
# =========================================================================

class TestPydanticContract:
    """Verify that env.step() returns a valid Reward model with a
    non-empty reason string and a float value."""

    def test_reward_is_pydantic_model(self):
        env = DevOpsWarRoomEnv()
        env.reset(task_id=Scenario.EASY)

        _, reward, _, _ = env.step(_make_raw_action("inspect database"))

        assert isinstance(reward, Reward)

    def test_reward_has_float_value(self):
        env = DevOpsWarRoomEnv()
        env.reset(task_id=Scenario.EASY)

        _, reward, _, _ = env.step(_make_raw_action("inspect database"))

        assert isinstance(reward.value, float)

    def test_reward_has_non_empty_reason(self):
        env = DevOpsWarRoomEnv()
        env.reset(task_id=Scenario.EASY)

        _, reward, _, _ = env.step(_make_raw_action("inspect database"))

        assert isinstance(reward.reason, str)
        assert len(reward.reason) > 0

    def test_reward_has_done_bool(self):
        env = DevOpsWarRoomEnv()
        env.reset(task_id=Scenario.EASY)

        _, reward, _, _ = env.step(_make_raw_action("inspect database"))

        assert isinstance(reward.done, bool)

    def test_observation_is_pydantic_model(self):
        env = DevOpsWarRoomEnv()
        obs = env.reset(task_id=Scenario.EASY)

        assert isinstance(obs, Observation)

    def test_step_returns_correct_tuple_shape(self):
        """step() must return (Observation, Reward, bool, dict)."""
        env = DevOpsWarRoomEnv()
        env.reset(task_id=Scenario.EASY)

        result = env.step(_make_raw_action("inspect database"))

        assert len(result) == 4
        obs, reward, done, info = result
        assert isinstance(obs, Observation)
        assert isinstance(reward, Reward)
        assert isinstance(done, bool)
        assert isinstance(info, dict)


# =========================================================================
# 6. Hard Task Solving
# =========================================================================

class TestHardTaskSolving:
    """Simulate the sequence: switch role dev -> rollback deploy database v2.3.0
    Assert: Reward is +0.40 and the database returns to UP.

    NOTE: The Hard scenario has `database: DEGRADED`. The Dev role is
    authorized to execute `rollback_deploy`. We rollback the `database`
    service since that is the only non-UP service in the Hard scenario.
    """

    def test_hard_task_solved_via_role_switch_and_rollback(self):
        env = DevOpsWarRoomEnv()
        obs = env.reset(task_id=Scenario.HARD)

        # Pre-conditions — database is DEGRADED
        assert obs.services["database"] == ServiceState.degraded

        # Step 1: Switch to Dev role  (costs 1 tick, reward = 0.0)
        obs_switch, reward_switch, _, _ = env.step(
            _make_raw_action("switch role Dev")
        )
        assert obs_switch.current_role == "Dev"
        assert reward_switch.value == pytest.approx(0.0)

        # Step 2: Rollback deploy on degraded service
        obs_fix, reward_fix, done, info = env.step(
            _make_raw_action("rollback deploy database v2.3.0")
        )

        # The base reward for correct diagnosis/rollback is +0.40
        assert reward_fix.value == pytest.approx(0.40), \
            f"Expected +0.40 for correct rollback, got {reward_fix.value}"

        # Database should be restored to UP
        assert obs_fix.services["database"] == ServiceState.healthy

    def test_hard_task_wrong_role_fails(self):
        """Without switching to Dev, SRE cannot rollback -> penalty."""
        env = DevOpsWarRoomEnv()
        env.reset(task_id=Scenario.HARD)

        _, reward, _, _ = env.step(
            _make_raw_action("rollback deploy database v2.3.0")
        )
        # SRE does not have rollback_deploy permission
        assert reward.value == pytest.approx(-0.15)

    def test_easy_task_direct_sre_restart(self):
        """Easy task: SRE can directly restart the database.
        Demonstrates difficulty progression."""
        env = DevOpsWarRoomEnv()
        obs = env.reset(task_id=Scenario.EASY)
        assert obs.services["database"] == ServiceState.down

        obs_fix, reward, _, _ = env.step(
            _make_raw_action("restart service database")
        )

        # +0.40 base for correct diagnosis
        assert reward.value == pytest.approx(0.40)
        assert obs_fix.services["database"] == ServiceState.healthy


# =========================================================================
# 7. OpenEnv BaseEnvironment Contract
# =========================================================================

class TestBaseEnvironmentContract:
    """Confirm DevOpsWarRoomEnv inherits from BaseEnvironment and
    exposes the required interface."""

    def test_inherits_base_environment(self):
        """Verify DevOpsWarRoomEnv inherits from whatever BaseEnvironment
        is resolved at import time (real openenv or fallback stub)."""
        mro = DevOpsWarRoomEnv.__mro__
        base_names = [cls.__name__ for cls in mro]
        assert "BaseEnvironment" in base_names, (
            f"Expected BaseEnvironment in MRO, got: {base_names}"
        )

    def test_has_reset_method(self):
        env = DevOpsWarRoomEnv()
        assert callable(getattr(env, "reset", None))

    def test_has_step_method(self):
        env = DevOpsWarRoomEnv()
        assert callable(getattr(env, "step", None))

    def test_reset_returns_observation(self):
        env = DevOpsWarRoomEnv()
        obs = env.reset(task_id=Scenario.EASY)
        assert isinstance(obs, Observation)

    def test_state_property_returns_dict(self):
        """env.state should be a property that returns the raw state dict."""
        env = DevOpsWarRoomEnv()
        env.reset(task_id=Scenario.EASY)
        state = env.state
        assert isinstance(state, dict)
        assert "services" in state
        assert "metrics" in state


# =========================================================================
# 8. Determinism & Isolation
# =========================================================================

class TestDeterminismAndIsolation:
    """Multiple resets with the same scenario must produce identical state."""

    def test_deterministic_reset(self):
        env = DevOpsWarRoomEnv()

        obs1 = env.reset(task_id=Scenario.EASY)
        obs2 = env.reset(task_id=Scenario.EASY)

        assert obs1.services == obs2.services
        assert obs1.tick == obs2.tick
        assert obs1.metrics == obs2.metrics
        assert len(obs1.alerts) == len(obs2.alerts)

    def test_independent_env_instances(self):
        """Two separate env instances must not share state."""
        env_a = DevOpsWarRoomEnv()
        env_b = DevOpsWarRoomEnv()

        env_a.reset(task_id=Scenario.EASY)
        env_b.reset(task_id=Scenario.MEDIUM)

        # Mutate env_a
        env_a.step(_make_raw_action("restart service database"))

        # env_b should be unaffected
        obs_b = env_b._get_observation()
        assert obs_b.services["database"] == ServiceState.healthy  # MEDIUM has DB UP
        assert obs_b.services["worker"] == ServiceState.degraded

    def test_noise_action_penalty_via_raw_command(self):
        """Unrecognizable raw commands are treated as unauthorized action_type='command',
        yielding -0.15 because 'command' is not in any role's permission set."""
        env = DevOpsWarRoomEnv()
        env.reset(task_id=Scenario.EASY)

        _, reward, _, _ = env.step(_make_raw_action("do a barrel roll"))
        assert reward.value == pytest.approx(-0.15)

    def test_noise_action_penalty_via_unknown_action_type(self):
        """The -0.05 noise penalty path in env.step() only fires when
        action_name resolves to the literal string 'unknown'. This happens
        when action.action_type is set to 'unknown' directly."""
        env = DevOpsWarRoomEnv()
        env.reset(task_id=Scenario.EASY)

        # action_type="unknown" → action_name="unknown" → -0.05 noise path
        _, reward, _, _ = env.step(
            Action(action_type="unknown")
        )
        assert reward.value == pytest.approx(-0.05)

    def test_unrecognized_action_type_hits_auth_check(self):
        """Any non-standard action_type that isn't literally 'unknown'
        falls through to the authorization check and gets -0.15."""
        env = DevOpsWarRoomEnv()
        env.reset(task_id=Scenario.EASY)

        _, reward, _, _ = env.step(
            Action(action_type="gibberish_xyz_123")
        )
        assert reward.value == pytest.approx(-0.15)


# =========================================================================
# Entry point for standalone execution
# =========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
