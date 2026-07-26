"""Low-fidelity, single-sentry tactical environment.

This is intentionally a tactical simulator, not a replacement for Gazebo.  It
models the consequences that matter to the high-level policy: reachability,
path risk, visibility, engagement, heat, objectives and reactive opponents.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

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
    NONE_TARGET = 3

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
        self.vector_dim = 7 + self.n_goals * 5 + 3 * 5 + 2 * 4
        self.reset(seed=seed)

    def reset(self, *, seed: int | None = None) -> dict[str, np.ndarray]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_count = 0

        def spawn(cell: Cell) -> Cell:
            return self.map.nearest_free(cell) or self.map.clamp_cell(cell)

        self.sentry = Unit(7, "red", spawn((3, 7)), hp=400.0, max_hp=400.0, ammo=260)
        self.allies = [
            Unit(3, "red", spawn((6, 5)), style="guard"),
            Unit(4, "red", spawn((6, 10)), style="support"),
        ]
        styles = self.rng.permutation(["pressure", "defend", "flank"])
        cells = [(22, 5), (23, 9), (19, 12)]
        self.enemies = [
            Unit(103 + i, "blue", spawn(cell), style=str(styles[i]))
            for i, cell in enumerate(cells)
        ]
        self.red_outpost_hp = 1500.0
        self.blue_outpost_hp = 1500.0
        self.active_goal_idx: int | None = None
        self.goal_lock_until = 0
        self.last_info: dict[str, Any] = {}
        return self.observe()

    def step(self, action: tuple[int, int, int] | list[int] | np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        requested_goal_idx, target_idx, fire_mode = (int(x) for x in action)
        reward_terms: dict[str, float] = {
            "time": -0.01,
            "invalid_action": 0.0,
            "damage_dealt": 0.0,
            "damage_taken": 0.0,
            "red_outpost_damage": 0.0,
            "blue_outpost_damage": 0.0,
            "red_outpost_control_loss": 0.0,
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

        self.sentry.heat = max(0.0, self.sentry.heat - 16.0 * self.decision_seconds)
        if fire_mode == self.FIRE_ENGAGE and target_idx != self.NONE_TARGET:
            damage = self._sentry_fire(target_idx)
            info["damage_dealt"] = damage
            reward_terms["damage_dealt"] = damage * 0.035
            reward += reward_terms["damage_dealt"]
        elif fire_mode not in (self.FIRE_HOLD, self.FIRE_ENGAGE):
            reward_terms["invalid_action"] -= 0.08
            reward += -0.08
            info["invalid_action"] = True

        # All non-sentry agents are merely reactive simulation participants.
        self._move_allies_and_attack()
        taken, red_outpost_damage = self._move_enemies_and_attack()
        info["damage_taken"] = taken
        info["red_outpost_damage"] = red_outpost_damage
        reward_terms["damage_taken"] = -taken * 0.030
        reward_terms["red_outpost_damage"] = -red_outpost_damage * 0.012
        reward += reward_terms["damage_taken"] + reward_terms["red_outpost_damage"]

        blue_delta, red_delta = self._update_outpost_control()
        reward_terms["blue_outpost_damage"] = blue_delta * 0.030
        reward += reward_terms["blue_outpost_damage"]
        reward_terms["red_outpost_control_loss"] = -red_delta * 0.018
        reward += reward_terms["red_outpost_control_loss"]
        info["blue_outpost_damage"] = blue_delta
        info["red_outpost_control_loss"] = red_delta
        if not self.sentry.alive:
            reward -= 8.0
            reward_terms["terminal"] -= 8.0
        self.step_count += 1
        done = (self.step_count >= self.horizon or not self.sentry.alive or
                self.red_outpost_hp <= 0.0 or self.blue_outpost_hp <= 0.0)
        if done:
            if self.blue_outpost_hp <= 0.0 and self.red_outpost_hp > 0.0:
                reward += 12.0
                reward_terms["terminal"] += 12.0
                info["outcome"] = "win"
            elif self.red_outpost_hp <= 0.0 or not self.sentry.alive:
                reward -= 10.0
                reward_terms["terminal"] -= 10.0
                info["outcome"] = "loss"
            else:
                info["outcome"] = "timeout"
        info.update(
            red_outpost_hp=self.red_outpost_hp,
            blue_outpost_hp=self.blue_outpost_hp,
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
        features: list[float] = [
            self.sentry.hp / self.sentry.max_hp,
            self.sentry.heat / 260.0,
            self.sentry.ammo / 260.0,
            self.red_outpost_hp / 1500.0,
            self.blue_outpost_hp / 1500.0,
            self.step_count / self.horizon,
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
        if not (0 <= target_idx < len(self.enemies)):
            return 0.0
        target = self.enemies[target_idx]
        if (not target.alive or not self._visible(target) or self.sentry.ammo <= 0 or
                self.sentry.heat > 235.0):
            return 0.0
        distance = self._distance(self.sentry.cell, target.cell)
        if distance > 8.0:
            return 0.0
        self.sentry.ammo -= 1
        self.sentry.heat += 15.0
        hit_probability = 0.83 * np.exp(-0.07 * distance)
        if self.rng.random() < hit_probability:
            damage = min(20.0, target.hp)
            target.hp -= damage
            return damage
        return 0.0

    def _move_allies_and_attack(self) -> None:
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
                    target.hp -= min(10.0, target.hp)

    def _move_enemies_and_attack(self) -> tuple[float, float]:
        sentry_damage, outpost_damage = 0.0, 0.0
        for enemy in self.enemies:
            if not enemy.alive:
                continue
            if enemy.style == "pressure":
                goal = self.map.red_outpost
            elif enemy.style == "defend":
                goal = self.map.blue_outpost
            else:
                goal = (15, 12) if enemy.cell[1] < 8 else (15, 2)
            self._move_towards(enemy, goal)
            if self.sentry.alive and self._distance(enemy.cell, self.sentry.cell) < 7.0 and self.map.line_of_sight(enemy.cell, self.sentry.cell):
                if self.rng.random() < 0.38:
                    damage = min(14.0, self.sentry.hp)
                    self.sentry.hp -= damage
                    sentry_damage += damage
            if self._distance(enemy.cell, self.map.red_outpost) < 2.5:
                damage = 5.0
                self.red_outpost_hp = max(0.0, self.red_outpost_hp - damage)
                outpost_damage += damage
        return sentry_damage, outpost_damage

    def _update_outpost_control(self) -> tuple[float, float]:
        red_units = [self.sentry, *self.allies]
        blue_units = self.enemies
        red_at_blue = sum(unit.alive and self._distance(unit.cell, self.map.blue_outpost) < 3.0 for unit in red_units)
        blue_at_blue = sum(unit.alive and self._distance(unit.cell, self.map.blue_outpost) < 3.0 for unit in blue_units)
        blue_delta = max(0.0, 6.0 * (red_at_blue - blue_at_blue))
        self.blue_outpost_hp = max(0.0, self.blue_outpost_hp - blue_delta)
        red_at_red = sum(unit.alive and self._distance(unit.cell, self.map.red_outpost) < 3.0 for unit in red_units)
        blue_at_red = sum(unit.alive and self._distance(unit.cell, self.map.red_outpost) < 3.0 for unit in blue_units)
        red_delta = max(0.0, 6.0 * (blue_at_red - red_at_red))
        self.red_outpost_hp = max(0.0, self.red_outpost_hp - red_delta)
        return blue_delta, red_delta

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
