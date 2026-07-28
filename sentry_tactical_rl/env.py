"""Low-fidelity, single-sentry tactical environment.

This is intentionally a tactical simulator, not a replacement for Gazebo.  It
models the consequences that matter to the high-level policy: reachability,
path risk, visibility, engagement, heat, objectives and reactive opponents.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .match_rules import MatchState, Team
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
    style: str = ""
    role: str = ""
    tunnel_defense_until_s: float = 0.0
    tunnel_cooling_until_s: float = 0.0

    @property
    def alive(self) -> bool:
        return self.hp > 0.0


class SentryTacticalEnv:
    """A Gym-like environment without a Gym dependency.

    The learned actor is only the red sentry.  Ally and enemy robots are
    reactive scripted agents for now; they are the seam where BC/IQL/DT policy
    adapters will later plug in.
    """

    FIRE_HOLD = 0
    FIRE_ENGAGE = 1
    ROBOT_TARGETS = 3
    BLUE_OUTPOST_TARGET = 3
    BLUE_BASE_TARGET = 4
    NONE_TARGET = 5

    def __init__(
        self,
        semantic_map: SemanticMap | None = None,
        *,
        horizon: int = 120,
        seed: int = 7,
        sensor_range: float = 13.0,
        decision_seconds: float = 1.0,
        goal_hold_seconds: float = 0.0,
    ) -> None:
        self.map = semantic_map or SemanticMap.demo()
        self.navigator = GridNavigationBackend(self.map)
        self.horizon = horizon
        self.sensor_range = sensor_range
        self.decision_seconds = decision_seconds
        self.goal_hold_seconds = max(0.0, float(goal_hold_seconds))
        self.goal_hold_steps = int(np.ceil(self.goal_hold_seconds / max(self.decision_seconds, 1e-6)))
        self.rng = np.random.default_rng(seed)
        self.anchor_names = self.map.anchor_names
        self.n_goals = len(self.anchor_names)
        self.n_targets = self.NONE_TARGET + 1
        self.map_channels = self.map.raster_base().shape[0] + 4
        # scalar + per-goal + per-enemy + per-ally feature layout
        self.vector_dim = 20 + self.n_goals * 5 + self.ROBOT_TARGETS * 5 + 2 * 4
        self.reset(seed=seed)

    def reset(self, *, seed: int | None = None) -> dict[str, np.ndarray]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_count = 0

        def spawn(cell: Cell) -> Cell:
            return self.map.nearest_free(cell) or self.map.clamp_cell(cell)

        self.sentry = Unit(7, "red", spawn((3, 7)), hp=400.0, max_hp=400.0, ammo=300, role="sentry")
        self.allies = [
            Unit(3, "red", spawn((6, 5)), style="guard", role="infantry3"),
            Unit(4, "red", spawn((6, 10)), style="support", role="infantry4"),
        ]
        styles = self.rng.permutation(["pressure", "defend", "flank"])
        cells = [(22, 5), (23, 9), (19, 12)]
        self.enemies = [
            Unit(103 + i, "blue", spawn(cell), style=str(styles[i]), role=("hero", "infantry3", "sentry")[i])
            for i, cell in enumerate(cells)
        ]
        self.match = MatchState(duration_s=self.horizon * self.decision_seconds)
        self.active_goal_idx: int | None = None
        self.goal_lock_until = 0
        self.rebuild_hold_s: dict[Team, float] = {"red": 0.0, "blue": 0.0}
        self._sentry_death_reported = False
        self._last_fire_result = {"robot": 0.0, "blue_outpost": 0.0, "blue_base": 0.0}
        self.last_info: dict[str, Any] = {}
        return self.observe()

    @property
    def red_outpost_hp(self) -> float:
        return self.match.red.outpost_hp

    @property
    def blue_outpost_hp(self) -> float:
        return self.match.blue.outpost_hp

    def step(self, action: tuple[int, int, int] | list[int] | np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        requested_goal_idx, target_idx, fire_mode = (int(x) for x in action)
        self._last_fire_result = {"robot": 0.0, "blue_outpost": 0.0, "blue_base": 0.0}
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
        }
        reward = reward_terms["time"]
        info: dict[str, Any] = {"invalid_action": False, "damage_dealt": 0.0, "damage_taken": 0.0}

        # The policy still proposes a discrete anchor in this demo.  A short
        # commitment window prevents rapid oscillation; continuous goal points
        # will use the same lock/ slew-rate interface when that action head is
        # introduced.
        goal_idx = requested_goal_idx
        goal_switch_blocked = False
        if (self.active_goal_idx is not None and goal_idx != self.active_goal_idx and
                self.step_count < self.goal_lock_until):
            goal_idx = self.active_goal_idx
            goal_switch_blocked = True
        goal_switched = goal_idx != self.active_goal_idx
        if goal_switched and 0 <= goal_idx < self.n_goals:
            self.active_goal_idx = goal_idx
            self.goal_lock_until = self.step_count + self.goal_hold_steps
            reward_terms["goal_switch"] = -0.02
            reward += reward_terms["goal_switch"]
        info["requested_goal_idx"] = requested_goal_idx
        info["executed_goal_idx"] = goal_idx
        info["goal_switch"] = goal_switched
        info["goal_switch_blocked"] = goal_switch_blocked
        threat = self._threat_layer(visible_only=False)
        path_cost = 0.0
        path_risk = 0.0

        # Goal execution is always delegated to the navigation backend.
        if 0 <= goal_idx < self.n_goals:
            goal = self.map.anchors[self.anchor_names[goal_idx]]
            nav = self.navigator.plan(self.sentry.cell, goal, threat)
            if nav.reachable:
                self._follow_path_one_step(self.sentry, nav.path)
                info["goal_name"] = self.anchor_names[goal_idx]
                info["path_cost"] = nav.path_cost
                path_cost = float(nav.path_cost)
                path_risk = self._path_risk(nav.path, threat)
            else:
                reward_terms["invalid_action"] -= 0.20
                reward += reward_terms["invalid_action"]
                info["invalid_action"] = True
                info["navigation"] = nav.reason
        else:
            reward_terms["invalid_action"] -= 0.20
            reward += reward_terms["invalid_action"]
            info["invalid_action"] = True

        self.sentry.heat = max(0.0, self.sentry.heat - self._heat_cooling_rate(self.sentry) * self.decision_seconds)
        if self.sentry.alive and fire_mode == self.FIRE_ENGAGE and target_idx != self.NONE_TARGET:
            damage = self._sentry_fire(target_idx)
            info["damage_dealt"] = damage
            reward_terms["damage_dealt"] = damage * 0.035
            reward += reward_terms["damage_dealt"]
        elif fire_mode not in (self.FIRE_HOLD, self.FIRE_ENGAGE):
            reward_terms["invalid_action"] -= 0.08
            reward += -0.08
            info["invalid_action"] = True

        # All non-sentry agents are merely reactive simulation participants.
        ally_damage = self._move_allies_and_attack()
        taken, red_outpost_damage, red_base_damage = self._move_enemies_and_attack()
        info["damage_taken"] = taken
        info["red_outpost_damage"] = red_outpost_damage
        info["red_base_damage"] = red_base_damage
        reward_terms["damage_taken"] = -taken * 0.030
        reward_terms["red_outpost_damage"] = -red_outpost_damage * 0.012
        reward_terms["red_base_damage"] = -red_base_damage * 0.020
        reward += (reward_terms["damage_taken"] + reward_terms["red_outpost_damage"] +
                   reward_terms["red_base_damage"])

        blue_outpost_damage = self._last_fire_result["blue_outpost"] + ally_damage["outpost"]
        blue_base_damage = self._last_fire_result["blue_base"] + ally_damage["base"]
        reward_terms["blue_outpost_damage"] = blue_outpost_damage * 0.030
        reward += reward_terms["blue_outpost_damage"]
        reward_terms["blue_base_damage"] = blue_base_damage * 0.025
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
        )
        self.last_info = info
        return self.observe(), float(reward), bool(done), info

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
        features: list[float] = [
            self.sentry.hp / self.sentry.max_hp,
            self.sentry.heat / 260.0,
            self.sentry.ammo / 300.0,
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
            nav = self.navigator.plan(self.sentry.cell, goal, threat)
            if nav.reachable:
                goal_mask[index] = True
                risk = self._path_risk(nav.path, threat)
                cost = min(nav.path_cost / 50.0, 2.0)
            else:
                risk, cost = 1.0, 2.0
            dist_red = self._distance(goal, self.map.red_outpost) / 30.0
            dist_blue = self._distance(goal, self.map.blue_outpost) / 30.0
            features.extend((float(nav.reachable), cost, risk, dist_red, dist_blue))

        # Keep the policy distribution consistent with the execution layer:
        # during a commitment window only the active goal is sampleable.  The
        # step() check remains as a defensive guard for external callers.
        if (self.active_goal_idx is not None and self.step_count < self.goal_lock_until and
                0 <= self.active_goal_idx < self.n_goals):
            goal_mask[:] = False
            goal_mask[self.active_goal_idx] = True

        target_mask = np.zeros(self.n_targets, dtype=bool)
        for index, enemy in enumerate(self.enemies):
            if visible[index] and enemy.alive:
                target_mask[index] = True
                dx = (enemy.cell[0] - self.sentry.cell[0]) / self.map.width
                dy = (enemy.cell[1] - self.sentry.cell[1]) / self.map.height
                distance = self._distance(self.sentry.cell, enemy.cell) / 30.0
                features.extend((dx, dy, enemy.hp / enemy.max_hp, distance, 1.0))
            else:
                features.extend((0.0, 0.0, 0.0, 1.0, 0.0))
        # Buildings are known map objects.  The base target remains masked in
        # this first policy version, while MatchState still enforces its legal
        # attackability for scripted agents and direct integration tests.
        target_mask[self.BLUE_OUTPOST_TARGET] = self.match.blue.outpost_alive
        target_mask[self.BLUE_BASE_TARGET] = False
        target_mask[self.NONE_TARGET] = True

        for ally in self.allies:
            dx = (ally.cell[0] - self.sentry.cell[0]) / self.map.width
            dy = (ally.cell[1] - self.sentry.cell[1]) / self.map.height
            features.extend((dx, dy, ally.hp / ally.max_hp, float(ally.alive)))
        vector = np.asarray(features, dtype=np.float32)
        if vector.size != self.vector_dim:
            raise RuntimeError(f"feature size {vector.size} != configured {self.vector_dim}")
        return vector, goal_mask, target_mask

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
            distance = np.hypot(xx - enemy.cell[0], yy - enemy.cell[1])
            out += np.exp(-distance / 4.0).astype(np.float32) * 0.55
        out[self.map.hard_blocked] = 0.0
        return np.clip(out, 0.0, 2.0)

    def _sentry_fire(self, target_idx: int) -> float:
        if not self.sentry.alive or self.sentry.ammo <= 0 or self.sentry.heat + 15.0 > 260.0:
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
        if distance > 8.0 or not self.map.line_of_sight(self.sentry.cell, target_cell):
            return 0.0
        self.sentry.ammo -= 1
        self.sentry.heat += 15.0
        hit_probability = 0.83 * np.exp(-0.07 * distance)
        if self.rng.random() >= hit_probability:
            return 0.0
        if target is not None:
            damage = self._deal_robot_damage(self.sentry, target, 20.0)
            self._last_fire_result["robot"] = damage
            return damage
        if building == "outpost":
            result = self.match.apply_outpost_damage("red", "blue", 20.0)
            self._last_fire_result["blue_outpost"] = result.applied
            return result.applied
        result = self.match.apply_base_damage("red", "blue", 20.0)
        self._last_fire_result["blue_base"] = result.applied
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
            if self._distance(ally.cell, target.cell) < 6.0 and self.map.line_of_sight(ally.cell, target.cell):
                if self.rng.random() < 0.42:
                    self._deal_robot_damage(ally, target, 10.0)
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
            if self.sentry.alive and self._distance(enemy.cell, self.sentry.cell) < 7.0 and self.map.line_of_sight(enemy.cell, self.sentry.cell):
                if self.rng.random() < 0.38:
                    damage = self._deal_robot_damage(enemy, self.sentry, 14.0)
                    sentry_damage += damage
            target_kind = "outpost" if self.match.red.outpost_alive else "base"
            target_cell = self._structure_cell("red", target_kind)
            if self._distance(enemy.cell, target_cell) <= 2.5 and self.map.line_of_sight(enemy.cell, target_cell):
                if target_kind == "outpost":
                    result = self.match.apply_outpost_damage("blue", "red", 5.0)
                    outpost_damage += result.applied
                else:
                    result = self.match.apply_base_damage("blue", "red", 5.0)
                    base_damage += result.applied
        return sentry_damage, outpost_damage, base_damage

    def _structure_cell(self, side: Team, kind: str) -> Cell:
        if kind not in ("outpost", "base"):
            raise ValueError(f"unknown structure kind: {kind}")
        cell = getattr(self.map, f"{side}_{kind}")
        return self.map.nearest_free(cell) or self.map.clamp_cell(cell)

    def _deal_robot_damage(self, attacker: Unit, target: Unit, raw_damage: float) -> float:
        multiplier = 1.0 - self._unit_defense_bonus(target)
        damage = min(float(round(raw_damage * multiplier)), target.hp)
        target.hp -= damage
        self.match.record_robot_damage(attacker.team, damage)
        return damage

    def _apply_semantic_effects(self) -> float:
        """Apply only effects whose geometry and rule values are known here."""
        sentry_healing = 0.0
        for unit in (self.sentry, *self.allies, *self.enemies):
            if not unit.alive:
                continue
            healing_regions = self.map.region_ids_at(unit.cell, kind="healing")
            if any(self.map.region_owner(region) == unit.team for region in healing_regions):
                before = unit.hp
                # The supplied ``healing`` geometry is treated as the team's
                # supply-zone healing area.  The later 25% out-of-combat mode
                # needs the referee's disengagement definition, so it is not
                # guessed in this tactical model.
                unit.hp = min(unit.max_hp, unit.hp + unit.max_hp * 0.10 * self.decision_seconds)
                if unit is self.sentry:
                    sentry_healing += unit.hp - before
        return sentry_healing

    def _unit_defense_bonus(self, unit: Unit) -> float:
        bonus = 0.0
        if self.map.has_semantic_kind(unit.cell, "central_highland"):
            bonus = max(bonus, 0.25)
        for region in self.map.region_ids_at(unit.cell, kind="fortress"):
            if self.map.region_owner(region) == unit.team and self.match.team(unit.team).outpost_destroyed_ever:
                bonus = max(bonus, 0.50)
        if unit.tunnel_defense_until_s > self.match.time_s:
            bonus = max(bonus, 0.50)
        return bonus

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
        # Current participants have no explicit engineer role.  Use the
        # official 10-second hero/infantry/sentry hold for the red sentry and
        # a representative blue ground unit; an engineer adapter can later
        # pass the official 5-second hold time through this same state machine.
        occupants: dict[Team, Unit | None] = {
            "red": self.sentry,
            "blue": next((enemy for enemy in self.enemies if enemy.alive), None),
        }
        for side, unit in occupants.items():
            if unit is None or not self.match.can_rebuild_outpost(side):
                self.rebuild_hold_s[side] = 0.0
                continue
            outpost_cell = self._structure_cell(side, "outpost")
            if unit.alive and self._distance(unit.cell, outpost_cell) <= 2.5:
                self.rebuild_hold_s[side] += self.decision_seconds
                if self.rebuild_hold_s[side] >= 10.0:
                    self.match.rebuild_outpost(side)
                    self.rebuild_hold_s[side] = 0.0
            else:
                self.rebuild_hold_s[side] = 0.0

    def _team_total_hp(self, side: Team) -> float:
        units = (self.sentry, *self.allies) if side == "red" else tuple(self.enemies)
        return float(sum(max(0.0, unit.hp) for unit in units))

    def _move_towards(self, unit: Unit, goal: Cell) -> None:
        nav = self.navigator.plan(unit.cell, goal, np.zeros_like(self.map.static_cost))
        if nav.reachable:
            self._follow_path_one_step(unit, nav.path)

    @staticmethod
    def _follow_path_one_step(unit: Unit, path: list[Cell]) -> None:
        if len(path) > 1:
            unit.cell = path[1]

    @staticmethod
    def _distance(a: Cell, b: Cell) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

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
