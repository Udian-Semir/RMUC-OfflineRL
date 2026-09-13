from simulation2d.resources import asset_path
from simulation2d.world.environment import SentryTacticalEnv
from simulation2d.world.semantic_map import SemanticMap


def formal_map() -> SemanticMap:
    return SemanticMap.from_aligned_json(
        asset_path("semantic_map_aligned.json"),
        obstacle_path=asset_path("blackwhite_map.png"),
    )


def test_interactive_world_advances_with_full_observation() -> None:
    env = SentryTacticalEnv(semantic_map=formal_map(), horizon=4, seed=17)
    observation = env.reset(seed=17)

    for _ in range(4):
        goal = int(observation["goal_mask"].nonzero()[0][0])
        target = int(observation["target_mask"].nonzero()[0][0])
        observation, _, done, info = env.step((goal, target, env.FIRE_ENGAGE))

    assert done
    assert observation["map"].shape == (15, 150, 280)
    assert observation["vector"].shape == (171,)
    assert "executed_target_idx" in info
    assert "goal_reached" in info


def test_mobile_target_replans_route_only_after_sentry_arrives() -> None:
    env = SentryTacticalEnv(semantic_map=SemanticMap.demo(), horizon=4, seed=17)
    observation = env.reset(seed=17)
    env.sentry.invulnerable_until_s = 999.0
    goal = int(observation["goal_mask"].nonzero()[0][0])
    target = env.enemies[2]
    target.cell = (20, 7)

    _, _, _, first = env.step((goal, 2, env.FIRE_HOLD))
    first_goal = first["execution_goal_cell"]
    target.cell = (16, 11)
    _, _, _, deferred = env.step((goal, 2, env.FIRE_HOLD))

    assert not deferred["goal_target_replan"]
    assert deferred["goal_target_replan_deferred"]
    assert deferred["goal_switch_blocked"]
    assert not deferred["concrete_goal_changed"]
    assert deferred["execution_goal_cell"] == first_goal

    env.sentry.cell = first_goal
    env.active_goal_reached = True
    target.cell = (16, 11)
    _, _, _, replanned = env.step((goal, 2, env.FIRE_HOLD))

    assert replanned["goal_target_replan"]
    assert not replanned["goal_target_replan_deferred"]
    assert replanned["concrete_goal_changed"]
    assert not replanned["goal_switch"]
    assert replanned["execution_goal_cell"] != first_goal


def test_controller_arrival_feedback_releases_current_route() -> None:
    env = SentryTacticalEnv(semantic_map=SemanticMap.demo(), horizon=4, seed=17)
    env.active_goal_cell = env.map.nearest_free((20, 7))
    env.active_goal_idx = 0
    env.active_goal_reached = False
    x_m, y_m = env.map.cell_to_meters(env.sentry.cell)

    env.set_external_sentry_state(
        time_s=1.0, x_m=x_m, y_m=y_m,
        hp=env.sentry.hp, max_hp=env.sentry.max_hp,
        goal_reached=True,
    )

    assert env.active_goal_reached


def test_scripted_backend_advances_ground_respawn_reader() -> None:
    env = SentryTacticalEnv(
        semantic_map=SemanticMap.demo(), horizon=4, seed=17,
        sparring_backend="scripted",
    )
    observation = env.reset(seed=17)
    unit = env.enemies[2]
    unit.hp = 0.0
    unit.respawn_remaining_s = 1.0
    goal = int(observation["goal_mask"].nonzero()[0][0])

    env.step((goal, env.NONE_TARGET, env.FIRE_HOLD))

    assert unit.alive
    assert unit.hp == unit.max_hp * 0.10


def test_offline_backend_refreshes_command_after_respawn() -> None:
    env = SentryTacticalEnv(
        semantic_map=SemanticMap.demo(), horizon=4, seed=17,
        sparring_backend="scripted",
    )
    unit = env.enemies[2]
    unit.hp = 0.0
    unit.respawn_remaining_s = 1.0
    refresh_saw_alive: list[bool] = []

    def record_refresh() -> None:
        refresh_saw_alive.append(unit.alive)

    env._sparring_commands.clear()
    env._refresh_offline_sparring_commands = record_refresh
    env._step_offline_sparring()

    assert unit.alive
    assert refresh_saw_alive == [True]
