from simulation2d.resources import asset_path
from simulation2d.runtime.live_engine import LiveSimulationEngine, PursuitPolicy
from simulation2d.world.environment import SentryTacticalEnv
from simulation2d.world.semantic_map import SemanticMap


def environment(horizon: int = 6) -> SentryTacticalEnv:
    semantic_map = SemanticMap.from_aligned_json(
        asset_path("semantic_map_aligned.json"),
        obstacle_path=asset_path("blackwhite_map.png"),
    )
    return SentryTacticalEnv(
        semantic_map=semantic_map,
        horizon=horizon,
        seed=23,
        sparring_backend="scripted",
    )


def test_pursuit_policy_drives_live_world_and_loops_episode() -> None:
    env = environment(horizon=3)
    engine = LiveSimulationEngine(env, PursuitPolicy(env), seed=23, loop_episodes=True)

    results = [engine.step() for _ in range(3)]

    assert results[-1].done
    assert engine.episode == 1
    assert engine.env.step_count == 0
    assert all(len(result.action) == 3 for result in results)
    assert all("executed_target_idx" in result.info for result in results)


def test_non_looping_engine_requires_reset_after_terminal() -> None:
    env = environment(horizon=1)
    engine = LiveSimulationEngine(env, PursuitPolicy(env), seed=23, loop_episodes=False)

    assert engine.step().done
    assert engine.finished


def test_external_sentry_state_updates_policy_observation() -> None:
    env = environment(horizon=3)
    engine = LiveSimulationEngine(env, PursuitPolicy(env), seed=23)
    requested_cell = env.map.nearest_free(env.map.meters_to_cell((10.0, 8.0)))
    assert requested_cell is not None
    requested_x, requested_y = env.map.cell_to_meters(requested_cell)

    engine.set_external_sentry_state(
        time_s=1.0, x_m=requested_x, y_m=requested_y,
        hp=env.sentry.hp, max_hp=env.sentry.max_hp,
        simulate_fire=True,
    )

    x_m, y_m = env.map.cell_to_meters(env.sentry.cell)
    assert abs(x_m - requested_x) <= env.map.resolution_m
    assert abs(y_m - requested_y) <= env.map.resolution_m
