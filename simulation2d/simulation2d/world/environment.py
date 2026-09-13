"""Low-fidelity, single-sentry tactical environment.

This is intentionally a tactical simulator, not a replacement for Gazebo.  It
models the consequences that matter to the high-level policy: reachability,
path risk, visibility, engagement, heat, objectives and reactive opponents.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .rules import MatchState, Team
from .navigation import GridNavigationBackend
from .semantic_map import Cell, SemanticMap


@dataclass
class Unit:
    unit_id: int
    team: str
    cell: Cell
    hp: float = 200.0
    max_hp: float = 200.0
    heat: float = 0.0
    ammo: int = 180
    max_ammo: int = 180
    shots_fired: int = 0
    style: str = ""
    role: str = ""
    weapon_damage: float = 10.0
    attack_range: float = 6.0
    hit_probability: float = 0.42
    heat_limit: float = 260.0
    heat_per_shot: float = 10.0
    shot_cooldown_s: float = 1.0
    next_fire_time_s: float = 0.0
    respawn_remaining_s: float = 0.0
    respawn_count: int = 0
    invulnerable_until_s: float = 0.0
    last_fire_time_s: float = -float("inf")
    last_damage_time_s: float = -float("inf")
    yaw_deg: float = 0.0
    power_w: float = 0.0
    observation_heat17_max: float = 260.0
    observation_heat42_max: float = 0.0
    tunnel_defense_until_s: float = 0.0
    tunnel_cooling_until_s: float = 0.0

    @property
    def alive(self) -> bool:
        return self.hp > 0.0 and self.respawn_remaining_s <= 0.0


@dataclass(frozen=True)
class TacticalUnitSpec:
    """Temporary tactical combat parameters for a fixed referee role.

    HP follows the official data schema.  The remaining values only keep the
    2D sparring world operational until referee weapon/heat data and physical
    validation replace them; they are not claims about official vehicle specs.
    """

    max_hp: float
    ammo: int
    weapon_damage: float
    attack_range: float
    hit_probability: float
    heat_limit: float = 260.0
    heat_per_shot: float = 10.0
    shot_cooldown_s: float = 1.0
    observation_heat17_max: float = 260.0
    observation_heat42_max: float = 0.0


ROLE_SPECS: dict[str, TacticalUnitSpec] = {
    # Observation heat limits are the representative referee telemetry values
    # used by the frozen offline policies.  They are deliberately separate
    # from the temporary combat heat limit above: the latter belongs to this
    # low-fidelity damage model, while the former must not fabricate an
    # impossible 260-point 17 mm barrel for a hero.
    # Official raw hit values: 17 mm = 20 and 42 mm = 200.  The tactical
    # world advances at 1 Hz; hero cadence is explicitly limited to 4 s.
    # A full 42 mm thermal model remains a later rule subsystem, so its
    # cadence is owned by shot_cooldown_s rather than the 17 mm heat field.
    "hero": TacticalUnitSpec(450.0, 200, 200.0, 6.5, 1.0, heat_limit=0.0, heat_per_shot=0.0, shot_cooldown_s=4.0, observation_heat17_max=50.0, observation_heat42_max=140.0),
    "engineer": TacticalUnitSpec(250.0, 0, 0.0, 0.0, 0.0, heat_limit=0.0, heat_per_shot=0.0, observation_heat17_max=0.0),
    "infantry3": TacticalUnitSpec(400.0, 180, 20.0, 5.0, 1.0, observation_heat17_max=180.0),
    "infantry4": TacticalUnitSpec(400.0, 180, 20.0, 5.0, 1.0, observation_heat17_max=180.0),
    "aerial": TacticalUnitSpec(100.0, 100, 20.0, 5.0, 1.0, observation_heat17_max=110.0),
    "sentry": TacticalUnitSpec(400.0, 300, 20.0, 8.0, 1.0, heat_per_shot=15.0),
}

# Effective, auto-aim-adjusted one-second damage for regular combat vehicles.
# These are deliberately separate from referee single-projectile damage.  The
# simulator makes one high-level decision each second, so it must not turn a
# target-locked 17 mm burst into a single 20 HP projectile.  Hero and sentry
# retain their explicit raw-shot cadence.
EFFECTIVE_DPS_ROLES = frozenset(("infantry3", "infantry4", "aerial"))
SENTRY_EFFECTIVE_DPS_ROLE = "sentry"
VEHICLE_CLOSE_RANGE_M = 3.0
VEHICLE_MID_RANGE_M = 5.0
VEHICLE_CLOSE_DAMAGE_PER_SECOND = 140.0
VEHICLE_MID_DAMAGE_PER_SECOND = 70.0
SENTRY_CLOSE_DAMAGE_PER_SECOND = 180.0
SENTRY_MID_DAMAGE_PER_SECOND = 100.0
STRUCTURE_DAMAGE_PER_SECOND = 200.0
OUTPOST_ATTACK_DISTANCE_M = 7.0
OUTPOST_ATTACK_GOAL_TOLERANCE_M = 0.35
REFEREE_ROLE_NAMES = {
    "hero": "英雄",
    "engineer": "工程",
    "infantry3": "步兵3",
    "infantry4": "步兵4",
    "aerial": "空中",
    "sentry": "哨兵",
}

# Median first valid referee position for each role/camp across the supplied
# official dataset. These replace the old hand-written demo formation.
OFFICIAL_SPAWN_M: dict[Team, dict[str, tuple[float, float]]] = {
    "red": {
        "sentry": (3.59, 8.14), "engineer": (1.98, 6.25),
        "infantry3": (3.13, 6.94), "infantry4": (2.73, 8.19),
        "aerial": (1.41, 13.39), "hero": (1.77, 8.52),
    },
    "blue": {
        "sentry": (24.42, 7.07), "engineer": (26.09, 8.86),
        "infantry3": (24.88, 8.07), "infantry4": (25.32, 7.47),
        "aerial": (26.70, 1.76), "hero": (26.30, 6.62),
    },
}

# These profiles are deliberately episode-constant.  They create a small,
# interpretable opponent-strength distribution without pretending that one
# frozen offline checkpoint already represents several learned route styles.
# The checkpoint still owns goal and target intent; the values below only
# control the probability that a legal learned engagement intent is executed
# by blue auto-aim.  They do not change the shared tactical damage rates.
BLUE_SPARRING_PROFILES: dict[str, float] = {
    "aggressive": 1.00,
    "pressure": 0.82,
    "measured": 0.62,
}


class SentryTacticalEnv:
    """A Gym-like environment without a Gym dependency.

    The learned actor is only the red sentry.  Ally and enemy robots are
    reactive scripted agents by default.  Set ``sparring_backend="offline"``
    to make every non-red-sentry slot execute a frozen IQL tactical sub-goal.
    """

    FIRE_HOLD = 0
    FIRE_ENGAGE = 1
    ROBOT_TARGETS = 6
    BLUE_OUTPOST_TARGET = 6
    BLUE_BASE_TARGET = 7
    NONE_TARGET = 8
    NONLEARNER_ALLIES = 5
    TARGET_FEATURE_DIM = 11
    RECOVERY_ENTER_HP_FRACTION = 0.20
    RECOVERY_RELEASE_HP_FRACTION = 0.90
    _DEFAULT_SPARRING_RUNS = {
        "hero": "hero_iql_tactical",
        "engineer": "engineer_iql_tactical",
        "infantry3": "infantry_iql_tactical",
        "infantry4": "infantry_iql_tactical",
        "aerial": "aerial_iql_tactical",
        "sentry": "blue_sentry_iql_tactical",
    }

    def __init__(
        self,
        semantic_map: SemanticMap | None = None,
        *,
        horizon: int = 120,
        seed: int = 7,
        sensor_range: float = 13.0,
        decision_seconds: float = 1.0,
        goal_hold_seconds: float = 0.0,
        goal_reach_tolerance_m: float = 0.35,
        sparring_backend: str = "scripted",
        sparring_update_seconds: float = 5.0,
        sparring_device: str = "cpu",
        sparring_fire_policy: str = "offline",
        blue_sparring_profiles: tuple[str, ...] | list[str] = ("aggressive",),
        sparring_run_dirs: Mapping[str, str] | None = None,
        blue_sparring_style_run_dirs: Mapping[str, Mapping[str, str]] | None = None,
    ) -> None:
        self.map = semantic_map or SemanticMap.demo()
        self.navigator = GridNavigationBackend(self.map)
        self.companion_map = self.map.companion_view()
        self.companion_navigator = GridNavigationBackend(self.companion_map)
        self.horizon = horizon
        self.sensor_range = sensor_range
        self.decision_seconds = decision_seconds
        self.goal_hold_seconds = max(0.0, float(goal_hold_seconds))
        self.goal_hold_steps = int(np.ceil(self.goal_hold_seconds / max(self.decision_seconds, 1e-6)))
        self.goal_reach_tolerance_m = max(0.05, float(goal_reach_tolerance_m))
        if sparring_backend not in {"scripted", "offline"}:
            raise ValueError("sparring_backend must be 'scripted' or 'offline'")
        if sparring_fire_policy not in {"offline", "legal_autoaim", "aggressive_blue", "intent_dps"}:
            raise ValueError(
                "sparring_fire_policy must be 'offline', 'legal_autoaim', 'aggressive_blue', or 'intent_dps'"
            )
        self.sparring_backend = sparring_backend
        # An offline checkpoint predicts whether a historical robot happened to
        # shoot in the *next one-second log sample*.  It is not a reliable
        # low-level trigger controller in a newly constructed world.  These
        # modes keep the learned sub-goal / target intent, then let the
        # simulator's range, line-of-sight, heat and ammunition checks decide
        # whether a legal shot can actually be taken.
        self.sparring_fire_policy = sparring_fire_policy
        self.blue_sparring_profiles = tuple(str(name) for name in blue_sparring_profiles)
        if not self.blue_sparring_profiles:
            raise ValueError("blue_sparring_profiles must contain at least one profile")
        unknown_profiles = set(self.blue_sparring_profiles) - set(BLUE_SPARRING_PROFILES)
        if unknown_profiles:
            raise ValueError(f"unknown blue sparring profiles: {sorted(unknown_profiles)}")
        self.sparring_update_seconds = max(float(sparring_update_seconds), self.decision_seconds)
        self.sparring_update_steps = max(1, int(round(self.sparring_update_seconds / self.decision_seconds)))
        self.sparring_device = sparring_device
        self.sparring_run_dirs = dict(self._DEFAULT_SPARRING_RUNS)
        if sparring_run_dirs is not None:
            self.sparring_run_dirs.update(sparring_run_dirs)
        self.blue_sparring_style_run_dirs = {
            str(profile): {str(role): str(run_dir) for role, run_dir in run_dirs.items()}
            for profile, run_dirs in (blue_sparring_style_run_dirs or {}).items()
        }
        unknown_style_profiles = set(self.blue_sparring_style_run_dirs) - set(BLUE_SPARRING_PROFILES)
        if unknown_style_profiles:
            raise ValueError(f"unknown blue style checkpoint profiles: {sorted(unknown_style_profiles)}")
        for profile, run_dirs in self.blue_sparring_style_run_dirs.items():
            missing_roles = set(ROLE_SPECS) - set(run_dirs)
            if missing_roles:
                raise ValueError(
                    f"blue style checkpoint profile {profile!r} is missing roles: {sorted(missing_roles)}"
                )
        self._sparring_policies: dict[tuple[str, str], Any] = {}
        self._sparring_policy_keys: dict[tuple[str, str], tuple[str, str]] = {}
        self._blue_style_policy_keys: dict[tuple[str, str], tuple[str, str]] = {}
        self._sparring_commands: dict[int, Any] = {}
        self._external_sentry_state: dict[str, float] | None = None
        self._external_sentry_simulate_fire = False
        self.rng = np.random.default_rng(seed)
        self.anchor_names = self.map.anchor_names
        self.n_goals = len(self.anchor_names)
        self.n_targets = self.NONE_TARGET + 1
        self.map_channels = self.map.raster_base().shape[0] + 4
        # scalar + per-goal + per-enemy + per-ally feature layout
        self.vector_dim = (
            20 + self.n_goals * 5 + self.ROBOT_TARGETS * self.TARGET_FEATURE_DIM + self.NONLEARNER_ALLIES * 4
        )
        if self.sparring_backend == "offline":
            self._load_offline_sparring_policies()
        self.reset(seed=seed)

    def reset(self, *, seed: int | None = None) -> dict[str, np.ndarray]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.blue_sparring_profile = str(self.rng.choice(self.blue_sparring_profiles))
        self.blue_autoaim_probability = BLUE_SPARRING_PROFILES[self.blue_sparring_profile]

        def spawn(cell: Cell) -> Cell:
            resolved = self.map.nearest_free(cell)
            if resolved is None:
                raise RuntimeError(f"no traversable spawn exists near {cell}")
            return resolved

        def spawn_m(x_m: float, y_m: float) -> Cell:
            return spawn(self.map.meters_to_cell((x_m, y_m)))

        def spawn_air_m(x_m: float, y_m: float) -> Cell:
            return self.map.meters_to_cell((x_m, y_m))

        red_spawn = OFFICIAL_SPAWN_M["red"]
        blue_spawn = OFFICIAL_SPAWN_M["blue"]
        self.sentry = self._spawn_unit(7, "red", spawn_m(*red_spawn["sentry"]), "sentry")
        self.allies = [
            self._spawn_unit(1, "red", spawn_m(*red_spawn["hero"]), "hero", style="guard"),
            self._spawn_unit(2, "red", spawn_m(*red_spawn["engineer"]), "engineer", style="support"),
            self._spawn_unit(3, "red", spawn_m(*red_spawn["infantry3"]), "infantry3", style="guard"),
            self._spawn_unit(4, "red", spawn_m(*red_spawn["infantry4"]), "infantry4", style="support"),
            self._spawn_unit(6, "red", spawn_air_m(*red_spawn["aerial"]), "aerial", style="flank"),
        ]
        styles = self.rng.permutation(["pressure", "defend", "flank", "pressure", "defend", "flank"])
        roles = ("hero", "engineer", "infantry3", "infantry4", "aerial", "sentry")
        ids = (101, 102, 103, 104, 106, 107)
        self.enemies = [
            self._spawn_unit(
                ids[i], "blue",
                (spawn_air_m(*blue_spawn[role]) if role == "aerial" else spawn_m(*blue_spawn[role])),
                role, style=str(styles[i]),
            )
            for i, role in enumerate(roles)
        ]
        self.match = MatchState(duration_s=self.horizon * self.decision_seconds)
        self.active_goal_idx: int | None = None
        self.active_goal_cell: Cell | None = None
        self.active_goal_reached = True
        self.goal_lock_until = 0
        self.rebuild_hold_s: dict[Team, float] = {"red": 0.0, "blue": 0.0}
        self.rebuild_holder_id: dict[Team, int | None] = {"red": None, "blue": None}
        self._sparring_commands.clear()
        self._external_sentry_state = None
        self._external_sentry_simulate_fire = False
        self._sentry_death_reported = False
        self._sentry_recovery_active = False
        self._central_highland_owner: Team | None = None
        self._last_fire_result = {"robot": 0.0, "blue_outpost": 0.0, "blue_base": 0.0}
        self._combat_events: list[dict[str, Any]] = []
        self._building_hold_state: dict[int, tuple[int, str, Cell, float]] = {}
        self.last_info: dict[str, Any] = {}
        return self.observe()

    @staticmethod
    def _spawn_unit(unit_id: int, team: Team, cell: Cell, role: str, *, style: str = "") -> Unit:
        spec = ROLE_SPECS[role]
        return Unit(
            unit_id=unit_id,
            team=team,
            cell=cell,
            hp=spec.max_hp,
            max_hp=spec.max_hp,
            ammo=spec.ammo,
            max_ammo=spec.ammo,
            style=style,
            role=role,
            weapon_damage=spec.weapon_damage,
            attack_range=spec.attack_range,
            hit_probability=spec.hit_probability,
            heat_limit=spec.heat_limit,
            heat_per_shot=spec.heat_per_shot,
            shot_cooldown_s=spec.shot_cooldown_s,
            observation_heat17_max=spec.observation_heat17_max,
            observation_heat42_max=spec.observation_heat42_max,
        )

    def _load_offline_sparring_policies(self) -> None:
        """Load each frozen checkpoint once per team/run pair.

        The red sentry remains the PPO-controlled entity.  The pooled infantry
        checkpoint is reused for both infantry slots through ``act_for``.
        """
        from ..agents.offline_sparring import OfflineSparringPolicy

        for team in ("red", "blue"):
            for role in ROLE_SPECS:
                if team == "red" and role == "sentry":
                    continue
                self._sparring_policy_keys[(team, role)] = self._load_sparring_policy(
                    OfflineSparringPolicy, team=team, role=role, configured=self.sparring_run_dirs[role]
                )

        # A real style profile swaps the blue frozen policy, not merely its
        # fire probability. Only profiles selectable in this environment are
        # loaded, which keeps one-profile replay jobs lightweight.
        for profile in self.blue_sparring_profiles:
            run_dirs = self.blue_sparring_style_run_dirs.get(profile)
            if run_dirs is None:
                continue
            for role in ROLE_SPECS:
                self._blue_style_policy_keys[(profile, role)] = self._load_sparring_policy(
                    OfflineSparringPolicy, team="blue", role=role, configured=run_dirs[role]
                )

    def _load_sparring_policy(self, policy_type: Any, *, team: Team, role: str,
                              configured: str) -> tuple[str, str]:
        requested = Path(configured)
        if requested.is_absolute():
            run_dir = requested
        else:
            candidates = (Path.cwd(), *Path(__file__).resolve().parents)
            repository_root = next(
                (root for root in candidates
                 if (root / "rm_runs").is_dir() and (root / "rm_rl").is_dir()),
                Path(__file__).resolve().parents[2],
            )
            direct = repository_root / requested
            run_dir = direct if direct.exists() else repository_root / "rm_runs" / requested
        if not run_dir.is_dir():
            raise FileNotFoundError(f"offline sparring checkpoint missing for {team}/{role}: {run_dir}")
        key = (team, str(run_dir.resolve()))
        if key not in self._sparring_policies:
            policy_map = self.map if role == "sentry" else self.companion_map
            self._sparring_policies[key] = policy_type(
                str(run_dir), semantic_map=policy_map, team=team, role=role, device=self.sparring_device)
        return key

    def _sparring_policy_for(self, unit: Unit) -> Any:
        try:
            key = (
                self._blue_style_policy_keys.get((self.blue_sparring_profile, unit.role))
                if unit.team == "blue" else None
            )
            return self._sparring_policies[key or self._sparring_policy_keys[(unit.team, unit.role)]]
        except KeyError as exc:  # pragma: no cover - protected by roster construction
            raise RuntimeError(f"no offline sparring policy for {unit.team}/{unit.role}") from exc

    @property
    def red_outpost_hp(self) -> float:
        return self.match.red.outpost_hp

    @property
    def blue_outpost_hp(self) -> float:
        return self.match.blue.outpost_hp

    def set_external_sentry_state(
        self, *, time_s: float, x_m: float, y_m: float, hp: float, max_hp: float,
        yaw_deg: float = 0.0, heat: float = 0.0, ammo17_fired: float = 0.0,
        power_w: float = 0.0, simulate_fire: bool = False,
        goal_reached: bool = False,
    ) -> None:
        """Inject an official recorded sentry state before one hybrid replay step."""
        self.match.time_s = float(np.clip(time_s, 0.0, self.match.duration_s))
        requested = self.map.meters_to_cell((x_m, y_m))
        resolved = self.map.nearest_free(requested)
        if resolved is None:
            raise RuntimeError("no traversable cell exists for external sentry state")
        self.sentry.cell = resolved
        self.sentry.max_hp = max(float(max_hp), 1.0)
        self.sentry.hp = float(np.clip(hp, 0.0, self.sentry.max_hp))
        self.sentry.yaw_deg = float(yaw_deg)
        self.sentry.heat = max(0.0, float(heat))
        self.sentry.shots_fired = max(0, int(round(ammo17_fired)))
        self.sentry.ammo = max(0, self.sentry.max_ammo - self.sentry.shots_fired)
        self.sentry.power_w = max(0.0, float(power_w))
        self._external_sentry_state = {"x_m": float(x_m), "y_m": float(y_m), "time_s": float(time_s)}
        self._external_sentry_simulate_fire = bool(simulate_fire)
        if self.active_goal_cell is not None:
            self.active_goal_reached = (
                bool(goal_reached)
                or self._distance(self.sentry.cell, self.active_goal_cell)
                <= self.goal_reach_tolerance_m
            )

    def clear_external_sentry_state(self) -> None:
        """Return control of sentry motion to the 2D navigation model."""
        self._external_sentry_state = None
        self._external_sentry_simulate_fire = False

    def step(self, action: tuple[int, int, int] | list[int] | np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        policy_goal_idx, policy_target_idx, policy_fire_mode = (int(x) for x in action)
        scheduled_goal_idx, target_idx, fire_mode = self._scheduled_sentry_action(
            policy_goal_idx, policy_target_idx, policy_fire_mode)
        self._last_fire_result = {"robot": 0.0, "blue_outpost": 0.0, "blue_base": 0.0}
        self._combat_events = []
        reward_terms: dict[str, float] = {
            "time": -0.01,
            "invalid_action": 0.0,
            "damage_dealt": 0.0,
            "damage_taken": 0.0,
            "red_outpost_damage": 0.0,
            "blue_outpost_damage": 0.0,
            "red_base_damage": 0.0,
            "blue_base_damage": 0.0,
            "healing": 0.0,
            "goal_switch": 0.0,
            "terminal": 0.0,
            "base_defense": 0.0,
            "defense_progress": 0.0,
            "target_search": 0.0,
            "pursuit_progress": 0.0,
            "navigation_cost": 0.0,
        }
        reward = reward_terms["time"]
        info: dict[str, Any] = {
            "invalid_action": False,
            "damage_dealt": 0.0,
            "damage_taken": 0.0,
            "blue_sparring_profile": self.blue_sparring_profile,
            "blue_autoaim_probability": self.blue_autoaim_probability,
            "policy_goal_idx": policy_goal_idx,
            "policy_target_idx": policy_target_idx,
            "policy_fire_mode": policy_fire_mode,
            "scheduled_goal_idx": scheduled_goal_idx,
            "executed_target_idx": target_idx,
            "executed_fire_mode": fire_mode,
        }

        # Every concrete route remains committed until arrival. A mobile
        # target may move while the sentry is travelling, but that only queues
        # the next standoff goal; it must not continuously replace the route
        # currently being executed by Gazebo.
        goal_idx = scheduled_goal_idx
        defense_pressure = self._base_defense_pressure()
        base_distance_before = self._distance(self.sentry.cell, self.map.red_base)
        active_hunt = self._active_hunt_phase()
        pursuit_target = self._sentry_ground_target(target_idx)
        target_distance_before = (
            self._distance(self.sentry.cell, pursuit_target.cell) if pursuit_target is not None else None
        )
        recovery_override = self._sentry_recovery_active
        outpost_override = (
            self.match.time_s >= 60.0
            and self.match.blue.outpost_alive
            and self._imminent_sentry_threat() is None
            and not recovery_override
        )
        commitment_active = self._goal_commitment_active()
        goal_switch_blocked = False
        if commitment_active and not (recovery_override or outpost_override):
            if self.active_goal_idx is not None:
                goal_idx = self.active_goal_idx
            goal_switch_blocked = scheduled_goal_idx != goal_idx

        selected_goal: Cell | None = None
        if 0 <= goal_idx < self.n_goals:
            selected_anchor = self.map.anchors[self.anchor_names[goal_idx]]
            selected_goal = self._sentry_execution_goal(selected_anchor, target_idx)

        target_goal_moved = (
            pursuit_target is not None
            and selected_goal is not None
            and self.active_goal_cell is not None
            and selected_goal != self.active_goal_cell
        )

        previous_goal = self.active_goal_cell
        keep_latched_goal = commitment_active and not (recovery_override or outpost_override)
        target_replan_deferred = target_goal_moved and keep_latched_goal
        target_replan = target_goal_moved and not keep_latched_goal
        if target_replan_deferred:
            goal_switch_blocked = True
        if keep_latched_goal and self.active_goal_cell is not None:
            goal = self.active_goal_cell
        else:
            goal = selected_goal
            if goal is not None:
                self.active_goal_idx = goal_idx
                self.active_goal_cell = goal
                self.active_goal_reached = self._distance(self.sentry.cell, goal) <= self.goal_reach_tolerance_m
                self.goal_lock_until = self.step_count + self.goal_hold_steps

        concrete_goal_changed = previous_goal is not None and goal is not None and goal != previous_goal
        anchor_switched = (
            concrete_goal_changed
            and not target_replan
            and not (recovery_override or outpost_override)
        )
        if anchor_switched:
            reward_terms["goal_switch"] = -0.02
            reward += reward_terms["goal_switch"]
        info["requested_goal_idx"] = policy_goal_idx
        info["executed_goal_idx"] = goal_idx
        info["goal_switch"] = anchor_switched
        info["concrete_goal_changed"] = concrete_goal_changed
        info["goal_target_replan"] = target_replan
        info["goal_target_replan_deferred"] = target_replan_deferred
        info["goal_switch_blocked"] = goal_switch_blocked
        info["goal_commitment_active"] = self._goal_commitment_active()
        info["goal_reached"] = self.active_goal_reached
        info["goal_switch_reason"] = (
            "recovery_override" if recovery_override else
            "outpost_override" if outpost_override else
            "target_replan" if target_replan else
            "target_replan_deferred" if target_replan_deferred else
            "reached" if not commitment_active else
            "committed"
        )
        threat = self._threat_layer(visible_only=False)
        path_cost = 0.0
        path_risk = 0.0
        external_sentry = self._external_sentry_state is not None
        if not external_sentry:
            self.sentry.power_w = 0.0

        if goal is not None:
            info["execution_goal_cell"] = goal
            info["execution_goal_xy_m"] = self.map.cell_to_meters(goal)

        # Goal execution is always delegated to the navigation backend.
        if external_sentry:
            info["sentry_external"] = True
            info["goal_name"] = "database_recorded"
        elif goal is not None:
            nav = self.navigator.plan(self.sentry.cell, goal, threat)
            if nav.reachable:
                self._follow_path_one_step(self.sentry, nav.path)
                info["goal_name"] = self.anchor_names[goal_idx]
                info["target_coupled"] = goal != selected_anchor
                info["path_cost"] = nav.path_cost
                path_cost = float(nav.path_cost)
                path_risk = self._path_risk(nav.path, threat)
                self.active_goal_reached = self._distance(self.sentry.cell, goal) <= self.goal_reach_tolerance_m
                reward_terms["navigation_cost"] = -0.001 * min(path_cost, 250.0)
                reward += reward_terms["navigation_cost"]
            else:
                reward_terms["invalid_action"] -= 0.20
                reward += reward_terms["invalid_action"]
                info["invalid_action"] = True
                info["navigation"] = nav.reason
                # A permanently unreachable commitment would deadlock the
                # policy.  Release it so the next observation can choose a
                # different reachable goal.
                self.active_goal_cell = None
                self.active_goal_reached = True
        else:
            reward_terms["invalid_action"] -= 0.20
            reward += reward_terms["invalid_action"]
            info["invalid_action"] = True

        info["goal_reached"] = self.active_goal_reached
        info["goal_commitment_active"] = self._goal_commitment_active()

        if not external_sentry or self._external_sentry_simulate_fire:
            self.sentry.heat = max(0.0, self.sentry.heat - self._heat_cooling_rate(self.sentry) * self.decision_seconds)
        if ((not external_sentry or self._external_sentry_simulate_fire)
                and self.sentry.alive and fire_mode == self.FIRE_ENGAGE
                and target_idx != self.NONE_TARGET):
            damage = self._sentry_fire(target_idx)
            info["damage_dealt"] = damage
            reward_terms["damage_dealt"] = damage * 0.035
            reward += reward_terms["damage_dealt"]
            if defense_pressure and 0 <= target_idx < self.ROBOT_TARGETS:
                target = self.enemies[target_idx]
                if target.unit_id in defense_pressure:
                    reward_terms["base_defense"] = damage * 0.050
                    reward += reward_terms["base_defense"]
        elif fire_mode not in (self.FIRE_HOLD, self.FIRE_ENGAGE):
            reward_terms["invalid_action"] -= 0.08
            reward += -0.08
            info["invalid_action"] = True

        # A healthy sentry is an active combat role.  Semantic gain areas may
        # remain valid intermediate goals, but are not an excuse to spend a
        # whole match without selecting, approaching, and engaging a vehicle.
        if active_hunt:
            if pursuit_target is None:
                reward_terms["target_search"] = -0.15
            elif fire_mode == self.FIRE_HOLD:
                reward_terms["target_search"] = -0.04
            reward += reward_terms["target_search"]
            if pursuit_target is not None and target_distance_before is not None:
                target_distance_after = self._distance(self.sentry.cell, pursuit_target.cell)
                progress = np.clip((target_distance_before - target_distance_after) / 2.0, -1.0, 1.0)
                reward_terms["pursuit_progress"] = float(0.12 * progress)
                reward += reward_terms["pursuit_progress"]

        # Every non-red-sentry role is either a deterministic smoke-test
        # participant or a frozen offlineRL policy whose five-second intent is
        # executed through the same grid navigation and fire checks.
        if self.sparring_backend == "offline":
            sparring = self._step_offline_sparring()
            ally_damage = {"outpost": sparring["blue_outpost_damage"], "base": sparring["blue_base_damage"]}
            taken = sparring["red_sentry_damage"]
            red_outpost_damage = sparring["red_outpost_damage"]
            red_base_damage = sparring["red_base_damage"]
            info["sparring_commands"] = len(self._sparring_commands)
        else:
            self._advance_respawns()
            for unit in self._sparring_units():
                self._cool_unit(unit)
            ally_damage = self._move_allies_and_attack()
            taken, red_outpost_damage, red_base_damage = self._move_enemies_and_attack()
        info["damage_taken"] = taken
        info["red_outpost_damage"] = red_outpost_damage
        info["red_base_damage"] = red_base_damage
        reward_terms["damage_taken"] = -taken * 0.030
        reward_terms["red_outpost_damage"] = -red_outpost_damage * 0.012
        reward_terms["red_base_damage"] = -red_base_damage * 0.040
        reward += (reward_terms["damage_taken"] + reward_terms["red_outpost_damage"] +
                   reward_terms["red_base_damage"])

        if defense_pressure and self.sentry.alive:
            base_distance_after = self._distance(self.sentry.cell, self.map.red_base)
            progress = max(-1.0, min(1.0, (base_distance_before - base_distance_after) / 10.0))
            reward_terms["defense_progress"] = 0.20 * progress
            reward += reward_terms["defense_progress"]

        blue_outpost_damage = self._last_fire_result["blue_outpost"] + ally_damage["outpost"]
        blue_base_damage = self._last_fire_result["blue_base"] + ally_damage["base"]
        # Keep learner and companion effects separate. Aggregate objective
        # damage only says that the world is interacting; it cannot establish
        # whether the PPO sentry caused the improvement.
        info["sentry_robot_damage"] = float(info["damage_dealt"])
        info["sentry_blue_outpost_damage"] = float(self._last_fire_result["blue_outpost"])
        info["sentry_blue_base_damage"] = float(self._last_fire_result["blue_base"])
        info["ally_blue_outpost_damage"] = float(ally_damage["outpost"])
        info["ally_blue_base_damage"] = float(ally_damage["base"])
        # PPO controls only the red sentry. Do not assign it positive credit
        # for frozen allies' damage; team losses remain collective penalties.
        reward_terms["blue_outpost_damage"] = self._last_fire_result["blue_outpost"] * 0.030
        reward += reward_terms["blue_outpost_damage"]
        reward_terms["blue_base_damage"] = self._last_fire_result["blue_base"] * 0.025
        reward += reward_terms["blue_base_damage"]
        info["blue_outpost_damage"] = blue_outpost_damage
        info["blue_base_damage"] = blue_base_damage

        healing = self._apply_semantic_effects()
        reward_terms["healing"] = healing * 0.003
        reward += reward_terms["healing"]
        self._update_rebuild_progress()

        if not self.sentry.alive and not self._sentry_death_reported:
            reward -= 8.0
            reward_terms["terminal"] -= 8.0
            self._sentry_death_reported = True
        self.match.advance(self.decision_seconds)
        self.step_count += 1
        done = self.match.is_terminal()
        if done:
            outcome = self.match.outcome(
                red_total_hp=self._team_total_hp("red"),
                blue_total_hp=self._team_total_hp("blue"),
            )
            info["outcome"] = outcome
            if outcome == "red_win":
                reward += 12.0
                reward_terms["terminal"] += 12.0
            elif outcome == "blue_win":
                reward -= 10.0
                reward_terms["terminal"] -= 10.0
        info.update(
            red_outpost_hp=self.red_outpost_hp,
            blue_outpost_hp=self.blue_outpost_hp,
            red_base_hp=self.match.red.base_hp,
            blue_base_hp=self.match.blue.base_hp,
            red_base_shield=self.match.red.base_shield,
            blue_base_shield=self.match.blue.base_shield,
            match_time_s=self.match.time_s,
            phase_flags=self.match.phase_flags,
            rebuild_hold_s=dict(self.rebuild_hold_s),
            path_cost=path_cost,
            path_risk=path_risk,
            total_cost=path_cost,
            reward_terms=reward_terms,
            combat_events=tuple(self._combat_events),
        )
        self.last_info = info
        return self.observe(), float(reward), bool(done), info

    def _scheduled_sentry_action(self, goal_idx: int, target_idx: int,
                                 fire_mode: int) -> tuple[int, int, int]:
        """Apply survival and one-minute outpost tactical skills."""
        if self._update_sentry_recovery_state():
            return self._sentry_supply_goal_index(), self.NONE_TARGET, self.FIRE_HOLD
        if self.match.time_s < 60.0 or not self.match.blue.outpost_alive:
            return goal_idx, target_idx, fire_mode
        imminent = self._imminent_sentry_threat()
        if imminent is not None:
            return goal_idx, self.enemies.index(imminent), self.FIRE_ENGAGE
        target_idx = self.BLUE_OUTPOST_TARGET
        if self.anchor_names:
            goal_idx = min(
                range(self.n_goals),
                key=lambda index: self._distance(
                    self.map.anchors[self.anchor_names[index]], self.map.blue_outpost))
        return goal_idx, target_idx, self.FIRE_ENGAGE

    def _imminent_sentry_threat(self) -> Unit | None:
        """Return the largest legal attacker if this second is lethal."""
        if not self.sentry.alive:
            return None
        attackers: list[tuple[float, Unit]] = []
        for enemy in self.enemies:
            if enemy.role == "aerial":
                continue
            if not self._can_fire(enemy) or not self._can_hit_unit(enemy, self.sentry):
                continue
            distance = self._distance(enemy.cell, self.sentry.cell)
            raw_damage, _ = self._robot_damage_for_tactical_second(enemy, distance)
            damage = raw_damage * (1.0 - self._unit_defense_bonus(self.sentry))
            if damage > 0.0:
                attackers.append((damage, enemy))
        if not attackers or sum(damage for damage, _ in attackers) < self.sentry.hp:
            return None
        return max(attackers, key=lambda item: item[0])[1]

    def _sentry_outpost_attack_goal(self) -> Cell:
        """Return the fixed 7 m firing point in the approach corridor.

        The blue outpost is elevated, so this designated lane has a legal
        high-target firing view even though the 2D ground occupancy raster
        contains the central obstacle between the cells.
        """
        outpost_xy = np.asarray(self.map.cell_to_meters(self.map.blue_outpost), dtype=np.float64)
        requested_xy = outpost_xy + np.asarray((-OUTPOST_ATTACK_DISTANCE_M, 0.0), dtype=np.float64)
        requested = self.map.meters_to_cell(tuple(requested_xy))
        candidates: list[tuple[float, Cell]] = []
        radius_cells = int(np.ceil((OUTPOST_ATTACK_DISTANCE_M + 0.5) / self.map.resolution_m))
        for y in range(max(0, requested[1] - radius_cells),
                       min(self.map.height, requested[1] + radius_cells + 1)):
            for x in range(max(0, requested[0] - radius_cells),
                           min(self.map.width, requested[0] + radius_cells + 1)):
                cell = (x, y)
                if not self.map.is_free(cell):
                    continue
                distance = self._distance(cell, self.map.blue_outpost)
                if distance < OUTPOST_ATTACK_DISTANCE_M - 0.5 or distance > OUTPOST_ATTACK_DISTANCE_M + 0.5:
                    continue
                score = 8.0 * self._distance(cell, requested) + abs(distance - OUTPOST_ATTACK_DISTANCE_M)
                candidates.append((score, cell))
        fallback = self.map.nearest_free(self.map.blue_outpost) or self.map.blue_outpost
        return min(candidates, default=(float("inf"), fallback), key=lambda item: item[0])[1]

    def _sentry_has_designated_outpost_view(self) -> bool:
        return self._distance(self.sentry.cell, self._sentry_outpost_attack_goal()) <= OUTPOST_ATTACK_GOAL_TOLERANCE_M

    def _update_sentry_recovery_state(self) -> bool:
        """Maintain a low-HP recovery skill with hysteresis."""
        if not self.sentry.alive:
            # Keep the skill latched through the respawn reader.  A ground
            # sentry returns at 10% HP and must not leave supply on the first
            # healing tick.
            self._sentry_recovery_active = True
        elif self._sentry_recovery_active:
            if self.sentry.hp >= self.sentry.max_hp * self.RECOVERY_RELEASE_HP_FRACTION:
                self._sentry_recovery_active = False
        elif self.sentry.hp < self.sentry.max_hp * self.RECOVERY_ENTER_HP_FRACTION:
            self._sentry_recovery_active = True
        return self._sentry_recovery_active

    def _sentry_supply_goal_index(self) -> int:
        supply = self._supply_cell("red")
        if supply is None:
            raise RuntimeError("red sentry recovery requires an annotated red supply zone")
        return min(
            range(self.n_goals),
            key=lambda index: self._distance(self.map.anchors[self.anchor_names[index]], supply),
        )

    def _goal_commitment_active(self) -> bool:
        """Whether the current concrete navigation goal is still committed.

        A one-second PPO action is allowed to change the target/fire intent,
        but not the destination while the sentry is travelling to it.  The
        minimum hold window prevents an immediate replan at the same position;
        the reach test is the actual release condition for longer paths.
        """
        return bool(
            self.active_goal_cell is not None
            and (
                not self.active_goal_reached
                or self.step_count < self.goal_lock_until
            )
        )

    def _active_hunt_phase(self) -> bool:
        """Return whether normal vehicle pursuit shaping is active."""
        return (
            self.sentry.alive
            and not self._update_sentry_recovery_state()
            and not (self.match.time_s >= 60.0 and self.match.blue.outpost_alive)
            and any(unit.alive and unit.role != "aerial" for unit in self.enemies)
        )

    def _sentry_ground_target(self, target_idx: int) -> Unit | None:
        if 0 <= target_idx < self.ROBOT_TARGETS:
            target = self.enemies[target_idx]
            if target.alive and target.role != "aerial":
                return target
        return None

    def _sentry_execution_goal(self, learned_goal: Cell, target_idx: int) -> Cell:
        """Couple a selected vehicle target to a reachable firing position."""
        if target_idx == self.BLUE_OUTPOST_TARGET and self.match.blue.outpost_alive:
            return self._sentry_outpost_attack_goal()
        target = self._sentry_ground_target(target_idx)
        if target is None or not self._active_hunt_phase():
            return learned_goal
        return self._attack_standoff_goal(self.sentry, target.cell, learned_goal)

    def observe(self) -> dict[str, np.ndarray]:
        visible = [self._visible(enemy) for enemy in self.enemies]
        threat = self._threat_layer(visible_only=True)
        raster = np.concatenate((self.map.raster_base(), self._entity_layers(visible), threat[None]), axis=0)
        vector, goal_mask, target_mask = self._vector_features(threat, visible)
        return {
            "map": raster.astype(np.float32),
            "vector": vector.astype(np.float32),
            "goal_mask": goal_mask,
            "target_mask": target_mask,
        }

    def _vector_features(self, threat: np.ndarray, visible: list[bool]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        phases = self.match.phase_flags
        recovery_active = self._update_sentry_recovery_state()
        features: list[float] = [
            self.sentry.hp / self.sentry.max_hp,
            self.sentry.heat / 260.0,
            self.sentry.ammo / max(self.sentry.max_ammo, 1),
            self.red_outpost_hp / 1500.0,
            self.blue_outpost_hp / 1500.0,
            self.match.red.base_hp / self.match.red.base_max_hp,
            self.match.blue.base_hp / self.match.blue.base_max_hp,
            self.match.red.base_shield / 150.0,
            self.match.blue.base_shield / 150.0,
            float(self.match.red.outpost_destroyed_ever),
            float(self.match.blue.outpost_destroyed_ever),
            min(self.match.time_s / self.match.duration_s, 1.0),
            float(phases["post_60_pressure"]),
            float(phases["large_energy_phase"]),
            float(phases["outpost_rebuild_closed"]),
            float(self.match.outpost_armor_rotating("red")),
            float(self.match.outpost_armor_rotating("blue")),
            self._unit_defense_bonus(self.sentry),
            min(self._heat_cooling_rate(self.sentry) / 105.0, 1.0),
            float(sum(unit.alive for unit in self.allies)) / len(self.allies),
        ]
        goal_mask = np.zeros(self.n_goals, dtype=bool)
        for index, name in enumerate(self.anchor_names):
            goal = self.map.anchors[name]
            # Planning every candidate at every 1 Hz PPO state multiplied a
            # 0.1 m A* search by 13. Actual reachability is still validated
            # when PPO selects a goal in step(); here use cheap local features
            # to score the action set.
            reachable = self.map.is_free(goal)
            goal_mask[index] = reachable
            risk = float(threat[goal[1], goal[0]]) if reachable else 1.0
            cost = min(self._distance(self.sentry.cell, goal) / 50.0, 2.0)
            dist_red = self._distance(goal, self.map.red_outpost) / 30.0
            dist_blue = self._distance(goal, self.map.blue_outpost) / 30.0
            features.extend((float(reachable), cost, risk, dist_red, dist_blue))

        # Keep the policy distribution consistent with the execution layer:
        # during a commitment window only the active goal is sampleable.  The
        # step() check remains as a defensive guard for external callers.
        if (self._goal_commitment_active()
                and self.active_goal_idx is not None
                and 0 <= self.active_goal_idx < self.n_goals):
            goal_mask[:] = False
            goal_mask[self.active_goal_idx] = True

        target_mask = np.zeros(self.n_targets, dtype=bool)
        defense_pressure = self._base_defense_pressure()
        candidate_threat = self._threat_layer(visible_only=False)
        for index, enemy in enumerate(self.enemies):
            # The radar station must be told which otherwise-distant unit is
            # attacking the base; without this, a defensive target is absent
            # from the action mask until the sentry happens to see it.
            base_attacker = enemy.unit_id in defense_pressure
            radar_known = enemy.alive and enemy.role != "aerial"
            if radar_known:
                target_mask[index] = True
                features.extend(self._target_candidate_features(enemy, candidate_threat))
            else:
                features.extend((0.0,) * self.TARGET_FEATURE_DIM)
        # Buildings are known map objects.  The base target remains masked in
        # this first policy version, while MatchState still enforces its legal
        # attackability for scripted agents and direct integration tests.
        target_mask[self.BLUE_OUTPOST_TARGET] = self.match.blue.outpost_alive
        target_mask[self.BLUE_BASE_TARGET] = False
        target_mask[self.NONE_TARGET] = True
        if recovery_active:
            goal_mask[:] = False
            goal_mask[self._sentry_supply_goal_index()] = True
            target_mask[:] = False
            target_mask[self.NONE_TARGET] = True
        elif self.match.time_s >= 60.0 and self.match.blue.outpost_alive:
            imminent = self._imminent_sentry_threat()
            target_mask[:] = False
            if imminent is None:
                target_mask[self.BLUE_OUTPOST_TARGET] = True
            else:
                target_mask[self.enemies.index(imminent)] = True

        for ally in self.allies:
            dx = (ally.cell[0] - self.sentry.cell[0]) / self.map.width
            dy = (ally.cell[1] - self.sentry.cell[1]) / self.map.height
            features.extend((dx, dy, ally.hp / ally.max_hp, float(ally.alive)))
        vector = np.asarray(features, dtype=np.float32)
        if vector.size != self.vector_dim:
            raise RuntimeError(f"feature size {vector.size} != configured {self.vector_dim}")
        return vector, goal_mask, target_mask

    def _target_candidate_features(self, enemy: Unit, threat: np.ndarray) -> tuple[float, ...]:
        """Structured per-vehicle cost/benefit features for the target head."""
        dx = (enemy.cell[0] - self.sentry.cell[0]) / self.map.width
        dy = (enemy.cell[1] - self.sentry.cell[1]) / self.map.height
        direct_distance = self._distance(self.sentry.cell, enemy.cell)
        standoff = self._attack_standoff_goal(self.sentry, enemy.cell, self.sentry.cell)
        nav = self.navigator.plan(self.sentry.cell, standoff, threat)
        reachable = float(nav.reachable)
        path_cost = min(nav.path_cost / 300.0, 3.0) if nav.reachable else 3.0
        path_risk = self._path_risk(nav.path, threat) if nav.reachable else 1.0
        standoff_distance = self._distance(standoff, enemy.cell)
        expected_damage, _ = self._robot_damage_for_tactical_second(self.sentry, standoff_distance)
        target_threat = min(enemy.weapon_damage / 200.0, 1.0)
        base_pressure = float(enemy.unit_id in self._base_defense_pressure())
        return (
            float(dx), float(dy), enemy.hp / max(enemy.max_hp, 1.0), min(direct_distance / 30.0, 2.0), 1.0,
            reachable, float(path_cost), float(path_risk), min(expected_damage / 180.0, 1.0),
            float(target_threat), base_pressure,
        )

    def _base_defense_pressure(self) -> dict[int, float]:
        """Return blue units currently threatening the red base.

        This is a radar-level alert, not a new blue policy.  It combines the
        frozen command intent with geometric proximity so the red sentry can
        learn an intercept decision before the first base damage tick.
        """
        if self.match.red.outpost_alive:
            return {}
        base = self._structure_cell("red", "base")
        pressure: dict[int, float] = {}
        for unit in self.enemies:
            if not unit.alive:
                continue
            command = self._sparring_commands.get(unit.unit_id)
            requested = str(getattr(command, "target_role", "") or "").lower()
            intent = requested in {"base", "基地", "red_base"}
            close = self._distance(unit.cell, base) <= unit.attack_range + 0.35
            if intent or close:
                pressure[unit.unit_id] = 1.0 if intent else 0.5
        return pressure

    def _entity_layers(self, visible: list[bool]) -> np.ndarray:
        # visible enemies, allies, and sentry position
        out = np.zeros((3, self.map.height, self.map.width), dtype=np.float32)
        for enemy, is_visible in zip(self.enemies, visible):
            if enemy.alive and is_visible:
                x, y = enemy.cell
                out[0, y, x] = enemy.hp / enemy.max_hp
        for ally in self.allies:
            if ally.alive:
                x, y = ally.cell
                out[1, y, x] = ally.hp / ally.max_hp
        x, y = self.sentry.cell
        out[2, y, x] = self.sentry.hp / self.sentry.max_hp
        return out

    def _visible(self, unit: Unit) -> bool:
        return unit.alive and self._distance(self.sentry.cell, unit.cell) <= self.sensor_range and self.map.line_of_sight(self.sentry.cell, unit.cell)

    def _threat_layer(self, *, visible_only: bool) -> np.ndarray:
        yy, xx = np.mgrid[0:self.map.height, 0:self.map.width]
        out = np.zeros((self.map.height, self.map.width), dtype=np.float32)
        for enemy in self.enemies:
            if not enemy.alive or (visible_only and not self._visible(enemy)):
                continue
            distance = np.hypot(xx - enemy.cell[0], yy - enemy.cell[1]) * self.map.resolution_m
            out += np.exp(-distance / 4.0).astype(np.float32) * 0.55
        out[self.map.hard_blocked] = 0.0
        return np.clip(out, 0.0, 2.0)

    def _sentry_fire(self, target_idx: int) -> float:
        if not self._can_fire(self.sentry):
            return 0.0
        target_cell: Cell
        target: Unit | None = None
        building: str | None = None
        if 0 <= target_idx < self.ROBOT_TARGETS:
            target = self.enemies[target_idx]
            if not target.alive or not self._visible(target):
                return 0.0
            target_cell = target.cell
        elif target_idx == self.BLUE_OUTPOST_TARGET and self.match.blue.outpost_alive:
            building = "outpost"
            target_cell = self._structure_cell("blue", building)
        elif target_idx == self.BLUE_BASE_TARGET:
            building = "base"
            target_cell = self._structure_cell("blue", building)
        else:
            return 0.0
        distance = self._distance(self.sentry.cell, target_cell)
        designated_outpost_view = (
            building == "outpost" and self._sentry_has_designated_outpost_view()
        )
        if distance > self.sentry.attack_range or (
                not self.map.line_of_sight(self.sentry.cell, target_cell)
                and not designated_outpost_view
        ):
            return 0.0
        if building is not None and not self._building_assault_ready(self.sentry, building):
            return 0.0
        self._consume_shot(self.sentry)
        if target is not None:
            raw_damage, band = self._robot_damage_for_tactical_second(self.sentry, distance)
            damage = self._deal_robot_damage(self.sentry, target, raw_damage)
            self._last_fire_result["robot"] = damage
            if damage > 0.0:
                self._record_combat_event(self.sentry, target, damage, distance, band)
            return damage
        if building == "outpost":
            result = self.match.apply_outpost_damage("red", "blue", self.sentry.weapon_damage)
            self._last_fire_result["blue_outpost"] = result.applied
            if result.applied > 0.0:
                self._record_combat_event(self.sentry, None, result.applied, distance, "raw_shot", structure="blue_outpost")
            return result.applied
        result = self.match.apply_base_damage("red", "blue", self.sentry.weapon_damage)
        self._last_fire_result["blue_base"] = result.applied
        if result.applied > 0.0:
            self._record_combat_event(self.sentry, None, result.applied, distance, "raw_shot", structure="blue_base")
        return result.applied

    def _move_allies_and_attack(self) -> dict[str, float]:
        building_damage = {"outpost": 0.0, "base": 0.0}
        for ally in self.allies:
            if not ally.alive:
                continue
            living = [enemy for enemy in self.enemies if enemy.alive]
            if not living:
                continue
            target = min(living, key=lambda enemy: self._distance(ally.cell, enemy.cell))
            # Guards stay close to red outpost until an enemy enters the middle.
            goal = target.cell if self._distance(target.cell, self.map.red_outpost) < 10.0 else self.map.red_outpost
            self._move_towards(ally, goal)
            self._fire_unit_at_robot(ally, target)
        return building_damage

    def _move_enemies_and_attack(self) -> tuple[float, float, float]:
        sentry_damage, outpost_damage, base_damage = 0.0, 0.0, 0.0
        for enemy in self.enemies:
            if not enemy.alive:
                continue
            if enemy.style == "pressure":
                goal = self.map.red_outpost if self.match.red.outpost_alive else self.map.red_base
            elif enemy.style == "defend":
                goal = self.map.blue_outpost
            else:
                goal = (15, 12) if enemy.cell[1] < 8 else (15, 2)
            self._move_towards(enemy, goal)
            if self.sentry.alive:
                sentry_damage += self._fire_unit_at_robot(enemy, self.sentry)
            building_result = {
                "red_sentry_damage": 0.0,
                "red_outpost_damage": 0.0,
                "red_base_damage": 0.0,
                "blue_outpost_damage": 0.0,
                "blue_base_damage": 0.0,
            }
            self._fire_unit_at_structure(enemy, building_result)
            outpost_damage += building_result["red_outpost_damage"]
            base_damage += building_result["red_base_damage"]
        return sentry_damage, outpost_damage, base_damage

    def _step_offline_sparring(self) -> dict[str, float]:
        """Execute frozen five-second intents without bypassing simulation safety."""
        # Let a unit whose respawn reader reaches zero join this tick's policy
        # snapshot instead of standing idle for one extra decision cycle.
        self._advance_respawns()
        if not self._sparring_commands or self.step_count % self.sparring_update_steps == 0:
            self._refresh_offline_sparring_commands()

        units = self._sparring_units()
        for unit in units:
            self._cool_unit(unit)
        # All commands come from one pre-movement snapshot.  Movement is then
        # applied before shooting so a Python iteration order cannot leak into
        # policy inference observations.
        for unit in units:
            if unit.alive and (command := self._sparring_commands.get(unit.unit_id)) is not None:
                self._move_sparring_unit(unit, self._sparring_execution_goal(unit, command))

        result = {
            "red_sentry_damage": 0.0,
            "red_outpost_damage": 0.0,
            "red_base_damage": 0.0,
            "blue_outpost_damage": 0.0,
            "blue_base_damage": 0.0,
        }
        for unit in units:
            if unit.alive and (command := self._sparring_commands.get(unit.unit_id)) is not None:
                self._execute_sparring_fire(unit, command, result)
        return result

    def _refresh_offline_sparring_commands(self) -> None:
        from ..agents.offline_sparring import from_tactical_env

        snapshot = from_tactical_env(self)
        self._sparring_commands = {
            unit.unit_id: self._sparring_policy_for(unit).act_for(snapshot, role=unit.role)
            for unit in self._sparring_units()
            if unit.alive
        }

    def _sparring_units(self) -> tuple[Unit, ...]:
        return tuple((*self.allies, *self.enemies))

    def _team_units(self, side: Team) -> tuple[Unit, ...]:
        return (self.sentry, *self.allies) if side == "red" else tuple(self.enemies)

    def _move_sparring_unit(self, unit: Unit, goal: Cell) -> None:
        unit.power_w = 0.0
        if unit.role == "aerial":
            self._fly_towards_goal(unit, goal)
            return
        navigator = self.navigator if unit.role == "sentry" else self.companion_navigator
        nav = navigator.plan(unit.cell, goal, self._team_threat_layer(unit.team))
        if nav.reachable:
            self._follow_path_one_step(unit, nav.path)

    def _fly_towards_goal(self, unit: Unit, goal: Cell) -> None:
        """Air unit follows its learned sub-goal directly, above ground obstacles."""
        dx, dy = goal[0] - unit.cell[0], goal[1] - unit.cell[1]
        distance = float(np.hypot(dx, dy))
        if distance <= 1e-6:
            return
        step = min(distance, max(1, int(round(1.0 / self.map.resolution_m))))
        next_cell = self.map.clamp_cell((
            int(round(unit.cell[0] + dx / distance * step)),
            int(round(unit.cell[1] + dy / distance * step)),
        ))
        unit.yaw_deg = math.degrees(math.atan2(next_cell[1] - unit.cell[1], next_cell[0] - unit.cell[0]))
        unit.cell = next_cell

    def _sparring_execution_goal(self, unit: Unit, command: Any) -> Cell:
        """Turn a learned target choice into an A*-reachable firing position.

        The offline policy owns *whether* and *what* to engage.  Its navigation
        head was trained independently, however, so it may name a target while
        emitting a five-second goal in another direction.  The classical
        executor resolves only that contradiction: a concrete, legal target
        gets a reachable stand-off point; every non-engagement decision keeps
        the learned navigation goal unchanged.
        """
        learned_goal = command.goal_cell
        if unit.weapon_damage <= 0.0:
            return learned_goal

        target = self._find_opponent_role(unit.team, command.target_role)
        if target is not None:
            # Ground fire cannot engage an aerial target in this simulator.
            if target.role == "aerial":
                return learned_goal
            if unit.role == "aerial":
                return self._aerial_standoff_goal(unit, target.cell)
            return self._attack_standoff_goal(unit, target.cell, learned_goal)

        kind = self._requested_structure(command.target_role)
        if kind is None:
            return learned_goal
        defender: Team = "blue" if unit.team == "red" else "red"
        defended = self.match.team(defender)
        # A dead outpost and a shielded base are not legal engagement targets.
        # Do not silently turn an obsolete outpost label into a base label.
        if (kind == "outpost" and not defended.outpost_alive) or (
                kind == "base" and defended.outpost_alive):
            return learned_goal
        if unit.role == "aerial":
            return self._aerial_structure_standoff_goal(
                unit, self._structure_cell(defender, kind))
        return self._attack_standoff_goal(
            unit, self._structure_cell(defender, kind), learned_goal)

    def _aerial_standoff_goal(self, unit: Unit, target_cell: Cell) -> Cell:
        """Keep an airborne target intent inside effective weapon range."""
        desired_cells = max(1.0, 2.5 / self.map.resolution_m)
        dx = float(unit.cell[0] - target_cell[0])
        dy = float(unit.cell[1] - target_cell[1])
        norm = float(np.hypot(dx, dy))
        if norm <= 1e-6:
            dx = -1.0 if unit.team == "red" else 1.0
            dy, norm = 0.0, 1.0
        return self.map.clamp_cell((
            int(round(target_cell[0] + dx / norm * desired_cells)),
            int(round(target_cell[1] + dy / norm * desired_cells)),
        ))

    def _aerial_structure_standoff_goal(self, unit: Unit, target_cell: Cell) -> Cell:
        """Use a stable team-facing orbit point while a structure is selected."""
        desired_cells = max(1, int(round(2.5 / self.map.resolution_m)))
        direction = -1 if unit.team == "red" else 1
        return self.map.clamp_cell((target_cell[0] + direction * desired_cells, target_cell[1]))

    def _attack_standoff_goal(self, unit: Unit, target_cell: Cell, fallback: Cell) -> Cell:
        """Pick a free visible cell inside the unit's real attack range."""
        if unit.attack_range <= 0.0:
            return fallback
        movement_map = self.map if unit.role == "sentry" else self.companion_map
        # Regular vehicles should close into the 0--3 m high-DPS ring. Hero
        # and sentry retain a little stand-off room for their raw-shot ranges.
        desired_range = 2.5 if self._role_uses_effective_dps(unit) else min(4.0, unit.attack_range * 0.65)
        candidates: list[tuple[float, Cell]] = []
        radius_cells = int(np.ceil(unit.attack_range / self.map.resolution_m))
        for y in range(max(0, target_cell[1] - radius_cells),
                       min(self.map.height, target_cell[1] + radius_cells + 1)):
            for x in range(max(0, target_cell[0] - radius_cells),
                           min(self.map.width, target_cell[0] + radius_cells + 1)):
                cell = (x, y)
                if not movement_map.is_free(cell) or cell == target_cell:
                    continue
                distance = self._distance(cell, target_cell)
                if distance < 0.75 or distance > unit.attack_range:
                    continue
                if not movement_map.line_of_sight(cell, target_cell):
                    continue
                score = 4.0 * abs(distance - desired_range) + self._distance(unit.cell, cell)
                candidates.append((score, cell))
        return min(candidates, default=(float("inf"), fallback), key=lambda item: item[0])[1]

    def _team_threat_layer(self, team: Team) -> np.ndarray:
        yy, xx = np.mgrid[0:self.map.height, 0:self.map.width]
        out = np.zeros((self.map.height, self.map.width), dtype=np.float32)
        for opponent in self._team_units("blue" if team == "red" else "red"):
            if not opponent.alive or opponent.weapon_damage <= 0.0:
                continue
            distance = np.hypot(xx - opponent.cell[0], yy - opponent.cell[1]) * self.map.resolution_m
            out += np.exp(-distance / 4.0).astype(np.float32) * 0.35
        out[self.map.hard_blocked] = 0.0
        return np.clip(out, 0.0, 2.0)

    def _execute_sparring_fire(self, unit: Unit, command: Any, result: dict[str, float]) -> None:
        if not self._can_fire(unit):
            return
        learned_gate = bool(command.fire_allowed)
        # New 12-D checkpoints name outpost/base explicitly. Legacy 10-D
        # checkpoints have no building class and retain the conservative
        # fire-gate + nearby-goal fallback for comparison only.
        requested_structure = self._requested_structure(command.target_role)
        structure_intent = requested_structure is not None
        legacy_structure_intent = (
            requested_structure is None
            and learned_gate
            and self._offline_structure_intent(unit, command)
        )
        target = self._find_opponent_role(unit.team, command.target_role)
        if target is not None and not self._can_hit_unit(unit, target):
            target = None
        reactive_target = (
            self._reactive_combat_target(unit)
            if target is None and not structure_intent and not legacy_structure_intent
            else None
        )
        learned_intent = structure_intent or legacy_structure_intent or command.target_role is not None
        reactive_intent = reactive_target is not None
        if not learned_intent and not reactive_intent:
            return
        autoaim_gate = self._sparring_autoaim_enabled(unit, command)
        if not learned_gate and not autoaim_gate and not reactive_intent:
            return

        target = target or reactive_target
        if target is not None:
            damage = self._fire_unit_at_robot(unit, target)
            if unit.team == "blue" and target is self.sentry:
                result["red_sentry_damage"] += damage
            return
        if structure_intent:
            self._fire_unit_at_structure(unit, result, requested_kind=requested_structure)
        elif legacy_structure_intent:
            self._fire_unit_at_structure(unit, result)

    def _reactive_combat_target(self, unit: Unit) -> Unit | None:
        """Low-level auto-aim target when an offline policy misses a close fight.

        Offline target labels are out of distribution in this constructed 2D
        world and frequently decode as ``none``.  Route and building intent
        remain offline-policy decisions, but a weapon-ready vehicle may not
        ignore a legal, nearby ground opponent.  The enemy sentry has highest
        local threat priority; otherwise use the nearest legal ground target.
        """
        if self.sparring_fire_policy != "intent_dps" or unit.weapon_damage <= 0.0:
            return None
        candidates = [
            opponent for opponent in self._team_units("blue" if unit.team == "red" else "red")
            if self._can_hit_unit(unit, opponent)
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda opponent: (0 if opponent.role == "sentry" else 1, self._distance(unit.cell, opponent.cell)),
        )

    def _sparring_autoaim_enabled(self, unit: Unit, command: Any) -> bool:
        """Whether a legal auto-aim controller may supplement a frozen gate."""
        if self.sparring_fire_policy == "offline":
            return False
        if self.sparring_fire_policy == "intent_dps":
            # Symmetric for red and blue: a frozen policy chooses the target;
            # range, line-of-sight and the combat model decide the result.
            return command.target_role is not None or self._offline_structure_intent(unit, command)
        if self.sparring_fire_policy == "legal_autoaim":
            # Preserve the learned target intention: a checkpoint that says
            # "no target" / no building approach does not become an
            # unconditional shooter.
            return command.target_role is not None or self._offline_structure_intent(unit, command)
        # The blue side may convert an offline *mobile-target* intent into a
        # legal auto-aim trigger.  It still cannot invent a target or fire at
        # a building without the offline fire-gate signal.
        return (
            unit.team == "blue"
            and (command.target_role is not None or self._offline_structure_intent(unit, command))
            and self.rng.random() < self.blue_autoaim_probability
        )

    def _offline_structure_intent(self, unit: Unit, command: Any) -> bool:
        """Infer a building engagement intent from the learned navigation goal."""
        if not hasattr(command, "goal_cell"):
            return False
        defender: Team = "blue" if unit.team == "red" else "red"
        kind = "outpost" if self.match.team(defender).outpost_alive else "base"
        return self._distance(command.goal_cell, self._structure_cell(defender, kind)) <= unit.attack_range

    @staticmethod
    def _requested_structure(target_role: str | None) -> str | None:
        if target_role in {"outpost", "前哨站"}:
            return "outpost"
        if target_role in {"base", "基地"}:
            return "base"
        return None

    def _can_hit_unit(self, attacker: Unit, target: Unit) -> bool:
        return (
            attacker.alive
            and target.alive
            and target.role != "aerial"
            and target.invulnerable_until_s <= self.match.time_s
            and self._distance(attacker.cell, target.cell) <= attacker.attack_range
            and self._has_fire_line(attacker, target.cell)
        )

    def _has_fire_line(self, attacker: Unit, target_cell: Cell) -> bool:
        # Air units fly and shoot above the 2D ground occupancy extrusion.
        return attacker.role == "aerial" or self.map.line_of_sight(attacker.cell, target_cell)

    def _find_opponent_role(self, team: Team, target_role: str | None) -> Unit | None:
        if target_role is None:
            return None
        for unit in self._team_units("blue" if team == "red" else "red"):
            if unit.alive and target_role in {unit.role, REFEREE_ROLE_NAMES[unit.role]}:
                return unit
        return None

    def _fire_unit_at_robot(self, attacker: Unit, target: Unit) -> float:
        if not self._can_fire(attacker) or not self._can_hit_unit(attacker, target):
            return 0.0
        distance = self._distance(attacker.cell, target.cell)
        self._consume_shot(attacker)
        raw_damage, band = self._robot_damage_for_tactical_second(attacker, distance)
        damage = self._deal_robot_damage(attacker, target, raw_damage)
        if damage > 0.0:
            self._record_combat_event(attacker, target, damage, distance, band)
        return damage

    def _fire_unit_at_structure(
        self,
        attacker: Unit,
        result: dict[str, float],
        *,
        requested_kind: str | None = None,
    ) -> None:
        if not self._can_fire(attacker):
            return
        defender: Team = "blue" if attacker.team == "red" else "red"
        kind = requested_kind or ("outpost" if self.match.team(defender).outpost_alive else "base")
        target_cell = self._structure_cell(defender, kind)
        distance = self._distance(attacker.cell, target_cell)
        if distance > attacker.attack_range or not self._has_fire_line(attacker, target_cell):
            return
        if not self._building_assault_ready(attacker, kind):
            return
        self._consume_shot(attacker)
        raw_damage, band = self._structure_damage_for_tactical_second(attacker)
        if kind == "outpost":
            damage = self.match.apply_outpost_damage(attacker.team, defender, raw_damage).applied
        else:
            damage = self.match.apply_base_damage(attacker.team, defender, raw_damage).applied
        result[f"{defender}_{kind}_damage"] += damage
        if damage > 0.0:
            self._record_combat_event(attacker, None, damage, distance, band, structure=f"{defender}_{kind}")

    def _building_assault_ready(self, attacker: Unit, kind: str) -> bool:
        """Require a clear two-second hold before continuous building DPS.

        A navigation/building label alone does not justify an immediate tower
        burst. The attacker must settle near the structure for two consecutive
        referee seconds and may not ignore a ground vehicle inside the 5 m
        engagement ring.
        """
        opponents_nearby = any(
            opponent.alive and opponent.role != "aerial"
            and self._distance(attacker.cell, opponent.cell) <= VEHICLE_MID_RANGE_M
            for opponent in self._team_units("blue" if attacker.team == "red" else "red")
        )
        previous = self._building_hold_state.get(attacker.unit_id)
        if (
            previous is None
            or previous[0] != self.step_count - 1
            or previous[1] != kind
            or previous[2] != attacker.cell
            or opponents_nearby
        ):
            hold_s = 0.0
        else:
            hold_s = previous[3] + self.decision_seconds
        self._building_hold_state[attacker.unit_id] = (self.step_count, kind, attacker.cell, hold_s)
        return hold_s >= 2.0

    @staticmethod
    def _role_uses_effective_dps(unit: Unit) -> bool:
        return unit.role in EFFECTIVE_DPS_ROLES or unit.role == SENTRY_EFFECTIVE_DPS_ROLE

    def _robot_damage_for_tactical_second(self, attacker: Unit, distance: float) -> tuple[float, str]:
        """Return target-locked vehicle damage for one tactical decision tick."""
        if not self._role_uses_effective_dps(attacker):
            return attacker.weapon_damage, "raw_shot"
        if distance <= VEHICLE_CLOSE_RANGE_M:
            damage = (SENTRY_CLOSE_DAMAGE_PER_SECOND
                      if attacker.role == SENTRY_EFFECTIVE_DPS_ROLE
                      else VEHICLE_CLOSE_DAMAGE_PER_SECOND)
            return damage * self.decision_seconds, "close_0_3m"
        if distance <= VEHICLE_MID_RANGE_M:
            damage = (SENTRY_MID_DAMAGE_PER_SECOND
                      if attacker.role == SENTRY_EFFECTIVE_DPS_ROLE
                      else VEHICLE_MID_DAMAGE_PER_SECOND)
            return damage * self.decision_seconds, "mid_3_5m"
        return 0.0, "out_of_effective_range"

    def _structure_damage_for_tactical_second(self, attacker: Unit) -> tuple[float, str]:
        if attacker.role in EFFECTIVE_DPS_ROLES:
            return STRUCTURE_DAMAGE_PER_SECOND * self.decision_seconds, "structure_dps"
        return attacker.weapon_damage, "raw_shot"

    def _record_combat_event(
        self,
        attacker: Unit,
        target: Unit | None,
        damage: float,
        distance: float,
        band: str,
        *,
        structure: str = "",
    ) -> None:
        self._combat_events.append({
            "attacker_id": attacker.unit_id,
            "attacker_team": attacker.team,
            "attacker_role": attacker.role,
            "target_id": target.unit_id if target is not None else None,
            "target_team": target.team if target is not None else "",
            "target_role": target.role if target is not None else structure,
            "damage": round(float(damage), 3),
            "distance_m": round(float(distance), 3),
            "band": band,
        })

    def _can_fire(self, unit: Unit) -> bool:
        return (
            unit.alive
            and unit.weapon_damage > 0.0
            and unit.ammo > 0
            and unit.next_fire_time_s <= self.match.time_s
            and (unit.heat_limit <= 0.0 or unit.heat + unit.heat_per_shot <= unit.heat_limit)
        )

    def _consume_shot(self, unit: Unit) -> None:
        unit.ammo -= 1
        unit.shots_fired += 1
        unit.next_fire_time_s = self.match.time_s + unit.shot_cooldown_s
        unit.last_fire_time_s = self.match.time_s
        if unit.heat_limit > 0.0:
            unit.heat += unit.heat_per_shot

    def _cool_unit(self, unit: Unit) -> None:
        if unit.heat_limit > 0.0:
            unit.heat = max(0.0, unit.heat - self._heat_cooling_rate(unit) * self.decision_seconds)

    def _structure_cell(self, side: Team, kind: str) -> Cell:
        if kind not in ("outpost", "base"):
            raise ValueError(f"unknown structure kind: {kind}")
        cell = getattr(self.map, f"{side}_{kind}")
        return self.map.nearest_free(cell) or self.map.clamp_cell(cell)

    def _deal_robot_damage(self, attacker: Unit, target: Unit, raw_damage: float) -> float:
        if target.role == "aerial" or not target.alive:
            return 0.0
        multiplier = 1.0 - self._unit_defense_bonus(target)
        damage = min(float(round(raw_damage * multiplier)), target.hp)
        target.hp -= damage
        if damage > 0.0:
            target.last_damage_time_s = self.match.time_s
        self.match.record_robot_damage(attacker.team, damage)
        if target.hp <= 0.0:
            target.hp = 0.0
            if target.role != "aerial":
                target.respawn_remaining_s = 10.0 + round(self.match.time_s / 10.0) + 20.0 * target.respawn_count
                target.respawn_count += 1
                target.heat = 0.0
        return damage

    def _advance_respawns(self) -> None:
        """Advance official-style ground-robot respawn readers.

        A dead ground robot returns through its own supply zone at 10% HP.
        The exact field interaction card is not in this low-fidelity map, so
        the nearest annotated healing region is the deterministic stand-in.
        Air support is intentionally excluded: the rulebook marks its damage
        and respawn mechanics as not applicable.
        """
        for unit in self._team_units("red") + self._team_units("blue"):
            if unit.role == "aerial" or unit.hp > 0.0 or unit.respawn_remaining_s <= 0.0:
                continue
            side = unit.team
            supply = self._supply_cell(side)
            boosted = supply is not None and self._distance(unit.cell, supply) <= 1.5
            if self.match.team(side).base_hp < 2000.0:
                boosted = True
            unit.respawn_remaining_s = max(
                0.0, unit.respawn_remaining_s - self.decision_seconds * (4.0 if boosted else 1.0)
            )
            if unit.respawn_remaining_s <= 0.0:
                if supply is None:
                    # Tiny synthetic unit-test maps predate semantic supply
                    # labels. Keep their deterministic fallback, but fail
                    # loudly for any real map that declares healing regions
                    # without an owner/center.
                    has_owned_supply = any(
                        kind == "healing" and self.map.region_owner(name) == side
                        for name, kind in self.map.region_kinds.items()
                    )
                    if has_owned_supply:
                        raise RuntimeError(
                            f"no healing/supply region is annotated for {side}; "
                            "refusing to respawn a unit at its base"
                        )
                    supply = self.map.nearest_free(self.map.red_base if side == "red" else self.map.blue_base)
                unit.cell = supply
                unit.hp = unit.max_hp * 0.10
                unit.heat = 0.0
                unit.invulnerable_until_s = self.match.time_s + 30.0

    def _supply_cell(self, side: Team) -> Cell | None:
        regions = [
            self.map.region_centers[name]
            for name, kind in self.map.region_kinds.items()
            if kind == "healing" and self.map.region_owner(name) == side
        ]
        if regions:
            return self.map.nearest_free(regions[0])
        return None

    def _apply_semantic_effects(self) -> float:
        """Apply only effects whose geometry and rule values are known here."""
        sentry_healing = 0.0
        for unit in self._team_units("red") + self._team_units("blue"):
            if not unit.alive:
                continue
            healing_regions = self.map.region_ids_at(unit.cell, kind="healing")
            if any(self.map.region_owner(region) == unit.team for region in healing_regions):
                before = unit.hp
                disengaged = self.match.time_s - max(unit.last_fire_time_s, unit.last_damage_time_s) >= 6.0
                heal_fraction = 0.25 if self.match.time_s >= 240.0 and disengaged else 0.10
                unit.hp = min(unit.max_hp, unit.hp + unit.max_hp * heal_fraction * self.decision_seconds)
                if unit is self.sentry:
                    sentry_healing += unit.hp - before
        return sentry_healing

    def _unit_defense_bonus(self, unit: Unit) -> float:
        bonus = 0.0
        if self.map.has_semantic_kind(unit.cell, "central_highland") and self._central_highland_owner_for_current_state() == unit.team:
            bonus = max(bonus, 0.25)
        for region in self.map.region_ids_at(unit.cell, kind="fortress"):
            if self.map.region_owner(region) == unit.team and self.match.team(unit.team).outpost_destroyed_ever:
                bonus = max(bonus, 0.50)
        if unit.tunnel_defense_until_s > self.match.time_s:
            bonus = max(bonus, 0.50)
        return bonus

    def _central_highland_owner_for_current_state(self) -> Team | None:
        """Apply the rulebook's mutually exclusive central-highland control."""
        occupants = {
            unit.team
            for unit in self._team_units("red") + self._team_units("blue")
            if unit.alive and unit.role != "aerial" and self.map.has_semantic_kind(unit.cell, "central_highland")
        }
        if len(occupants) == 1:
            self._central_highland_owner = next(iter(occupants))
        elif not occupants:
            self._central_highland_owner = None
        elif self._central_highland_owner not in occupants:
            # Simultaneous first arrival is contested and grants no side the
            # benefit until one side leaves; an existing owner retains it.
            self._central_highland_owner = None
        return self._central_highland_owner

    def _heat_cooling_rate(self, unit: Unit) -> float:
        rate = 30.0
        for region in self.map.region_ids_at(unit.cell, kind="fortress"):
            team_state = self.match.team(unit.team)
            if self.map.region_owner(region) == unit.team and team_state.outpost_destroyed_ever:
                delta = team_state.base_max_hp - team_state.base_lowest_hp
                rate = max(rate, 30.0 + min(75.0, float(int(delta // 40.0))))
        if unit.tunnel_cooling_until_s > self.match.time_s:
            rate = max(rate, 60.0)
        return rate

    def activate_tunnel_gain_after_valid_sequence(self, unit: Unit) -> None:
        """Apply the official tunnel bonus after an external ordered-card check.

        The current delivered JSON has tunnel polygons but no sequence/group
        metadata, so entering one cell alone must not create a false bonus.
        A future semantic export or referee event validates the required
        end--middle--end traversal and then calls this method.
        """
        if not self.map.has_semantic_kind(unit.cell, "tunnel_gain"):
            raise ValueError("tunnel bonus requires the unit to be in a tunnel gain region")
        unit.tunnel_defense_until_s = self.match.time_s + 10.0
        unit.tunnel_cooling_until_s = self.match.time_s + 120.0

    def _update_rebuild_progress(self) -> None:
        for side in ("red", "blue"):
            if not self.match.can_rebuild_outpost(side):
                self.rebuild_hold_s[side] = 0.0
                self.rebuild_holder_id[side] = None
                continue
            outpost_cell = self._structure_cell(side, "outpost")
            candidates = [
                unit for unit in self._team_units(side)
                if unit.alive and self._distance(unit.cell, outpost_cell) <= 2.5
            ]
            if not candidates:
                self.rebuild_hold_s[side] = 0.0
                self.rebuild_holder_id[side] = None
                continue
            unit = min(candidates, key=lambda item: self._distance(item.cell, outpost_cell))
            if self.rebuild_holder_id[side] != unit.unit_id:
                self.rebuild_holder_id[side] = unit.unit_id
                self.rebuild_hold_s[side] = 0.0
            self.rebuild_hold_s[side] += self.decision_seconds
            required_hold = 5.0 if unit.role == "engineer" else 10.0
            if self.rebuild_hold_s[side] >= required_hold:
                self.match.rebuild_outpost(side)
                self.rebuild_hold_s[side] = 0.0
                self.rebuild_holder_id[side] = None

    def _team_total_hp(self, side: Team) -> float:
        return float(sum(max(0.0, unit.hp) for unit in self._team_units(side)))

    def _move_towards(self, unit: Unit, goal: Cell) -> None:
        unit.power_w = 0.0
        navigator = self.navigator if unit.role == "sentry" else self.companion_navigator
        nav = navigator.plan(unit.cell, goal, np.zeros_like(self.map.static_cost))
        if nav.reachable:
            self._follow_path_one_step(unit, nav.path)

    def _follow_path_one_step(self, unit: Unit, path: list[Cell]) -> None:
        if len(path) > 1:
            previous = unit.cell
            # Preserve the original simulator's 1 m/s tactical traversal while
            # allowing the planner itself to operate at decimetre resolution.
            steps = max(1, int(round(1.0 / self.map.resolution_m)))
            unit.cell = path[min(steps, len(path) - 1)]
            dx, dy = unit.cell[0] - previous[0], unit.cell[1] - previous[1]
            unit.yaw_deg = math.degrees(math.atan2(dy, dx))
            # A moving ground chassis should not be represented as permanently
            # idle in the referee feature contract.  The official aerial slot
            # has no comparable chassis-power telemetry; feeding it 100 W puts
            # every other policy's aerial feature hundreds of standard
            # deviations out of distribution.
            unit.power_w = 0.0 if unit.role == "aerial" else 100.0

    def _distance(self, a: Cell, b: Cell) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]) * self.map.resolution_m)

    @staticmethod
    def _path_risk(path: list[Cell], threat: np.ndarray) -> float:
        if not path:
            return 1.0
        return float(np.mean([threat[y, x] for x, y in path]))

    def render_ascii(self) -> str:
        """Small text renderer useful before any visualizer exists."""
        canvas = np.full((self.map.height, self.map.width), ".", dtype="U1")
        canvas[self.map.hard_blocked] = "#"
        for x, y in (self.map.red_outpost, self.map.blue_outpost):
            canvas[y, x] = "O"
        for enemy in self.enemies:
            if enemy.alive:
                x, y = enemy.cell
                canvas[y, x] = "E"
        for ally in self.allies:
            if ally.alive:
                x, y = ally.cell
                canvas[y, x] = "A"
        x, y = self.sentry.cell
        canvas[y, x] = "S"
        return "\n".join("".join(row) for row in canvas)
