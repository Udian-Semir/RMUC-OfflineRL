"""Deterministic RMUC 2026 match state used by the tactical environment.

This module owns building state and match resolution.  It deliberately does
not simulate every referee-system subsystem yet, but the rules represented
here are kept separate from robot motion so they can be regression tested and
reused by a future Gazebo/ROS adapter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


Team = Literal["red", "blue"]


@dataclass
class DamageResult:
    """Result of one legal or rejected building damage attempt."""

    applied: float = 0.0
    shield_damage: float = 0.0
    hp_damage: float = 0.0
    blocked_reason: str | None = None

    @property
    def accepted(self) -> bool:
        return self.applied > 0.0


@dataclass
class TeamMatchState:
    """Rule-relevant state for one side of a single match."""

    base_hp: float = 5000.0
    base_max_hp: float = 5000.0
    base_shield: float = 150.0
    base_lowest_hp: float = 5000.0
    cumulative_base_hp_loss: float = 0.0
    outpost_hp: float = 1500.0
    outpost_max_hp: float = 1500.0
    outpost_destroyed_ever: bool = False
    rebuilds_used: int = 0
    attack_damage: float = 0.0
    base_armor_deployed: bool = False

    @property
    def outpost_alive(self) -> bool:
        return self.outpost_hp > 0.0

    @property
    def rebuild_opportunities_earned(self) -> int:
        # The rule grants an accumulated opportunity per 1000 lost *base HP*.
        # Virtual-shield absorption is intentionally excluded.
        return int(self.cumulative_base_hp_loss // 1000.0)

    @property
    def rebuild_opportunities_available(self) -> int:
        return max(0, self.rebuild_opportunities_earned - self.rebuilds_used)


@dataclass
class MatchState:
    """Minimal, deterministic implementation of RMUC 2026 building rules.

    Scope deliberately covered here:

    * 5000 HP base with a 150 virtual shield;
    * base invulnerability while its outpost is alive;
    * 1500 HP outpost, accumulated rebuild opportunities, and the 300 s cutoff;
    * 60/180/300 s tactical phase flags and outpost armour rotation state;
    * official terminal and tie-break ordering.

    Damage multipliers, dart mechanics, economy, and physical hit detection are
    supplied by callers.  The ``amount`` passed here is therefore already the
    post-hit, post-defence attack damage.
    """

    duration_s: float = 420.0
    time_s: float = 0.0
    red: TeamMatchState = field(default_factory=TeamMatchState)
    blue: TeamMatchState = field(default_factory=TeamMatchState)

    def __post_init__(self) -> None:
        if self.duration_s <= 0.0:
            raise ValueError("duration_s must be positive")

    def team(self, side: Team) -> TeamMatchState:
        if side == "red":
            return self.red
        if side == "blue":
            return self.blue
        raise ValueError(f"unknown team: {side!r}")

    @staticmethod
    def opponent(side: Team) -> Team:
        if side == "red":
            return "blue"
        if side == "blue":
            return "red"
        raise ValueError(f"unknown team: {side!r}")

    def advance(self, seconds: float) -> None:
        if seconds < 0.0:
            raise ValueError("match time cannot move backwards")
        self.time_s += float(seconds)

    @property
    def phase_flags(self) -> dict[str, bool]:
        """Expose time-derived state without encoding it in reward weights."""
        return {
            # This is the team's agreed tactical pressure phase, not an
            # official mandatory action.
            "post_60_pressure": self.time_s >= 60.0,
            "large_energy_phase": self.time_s >= 180.0,
            "outpost_rotation_stopped_by_time": self.time_s >= 180.0,
            "outpost_rebuild_closed": self.time_s >= 300.0,
        }

    def outpost_armor_rotating(self, side: Team) -> bool:
        state = self.team(side)
        return (
            state.outpost_alive
            and not state.outpost_destroyed_ever
            and not state.base_armor_deployed
            and self.time_s < 180.0
        )

    def apply_outpost_damage(self, attacker: Team, defender: Team, amount: float) -> DamageResult:
        self._validate_sides(attacker, defender)
        amount = self._validate_damage(amount)
        target = self.team(defender)
        if not target.outpost_alive:
            return DamageResult(blocked_reason="outpost_destroyed")
        applied = min(amount, target.outpost_hp)
        target.outpost_hp -= applied
        self.team(attacker).attack_damage += applied
        if target.outpost_hp <= 0.0:
            target.outpost_hp = 0.0
            target.outpost_destroyed_ever = True
        return DamageResult(applied=applied, hp_damage=applied)

    def apply_base_damage(self, attacker: Team, defender: Team, amount: float) -> DamageResult:
        self._validate_sides(attacker, defender)
        amount = self._validate_damage(amount)
        target = self.team(defender)
        if target.outpost_alive:
            return DamageResult(blocked_reason="base_protected_by_outpost")
        if target.base_hp <= 0.0:
            return DamageResult(blocked_reason="base_destroyed")

        shield_damage = min(amount, target.base_shield)
        target.base_shield -= shield_damage
        hp_damage = min(amount - shield_damage, target.base_hp)
        target.base_hp -= hp_damage
        target.base_lowest_hp = min(target.base_lowest_hp, target.base_hp)
        target.cumulative_base_hp_loss += hp_damage
        applied = shield_damage + hp_damage
        # Shield absorption still counts as the attacker's total attack damage.
        self.team(attacker).attack_damage += applied
        return DamageResult(applied=applied, shield_damage=shield_damage, hp_damage=hp_damage)

    def record_robot_damage(self, attacker: Team, amount: float) -> float:
        """Record valid robot attack damage for the official final tie-break."""
        amount = self._validate_damage(amount)
        self.team(attacker).attack_damage += amount
        return amount

    def can_rebuild_outpost(self, side: Team) -> bool:
        state = self.team(side)
        return (
            not state.outpost_alive
            and self.time_s < 300.0
            and state.rebuild_opportunities_available > 0
        )

    def rebuild_outpost(self, side: Team) -> bool:
        """Apply a completed 10 s (or engineer 5 s) occupation externally."""
        if not self.can_rebuild_outpost(side):
            return False
        state = self.team(side)
        state.outpost_hp = 750.0
        state.rebuilds_used += 1
        return True

    def is_terminal(self) -> bool:
        return self.red.base_hp <= 0.0 or self.blue.base_hp <= 0.0 or self.time_s >= self.duration_s

    def winner(self, *, red_total_hp: float, blue_total_hp: float) -> Team | None:
        """Return the official winner, or ``None`` for pending/draw states."""
        if not self.is_terminal():
            return None
        result = self._higher(self.red.base_hp, self.blue.base_hp)
        if result is not None:
            return result

        # The remaining comparisons apply at time expiry.  If both bases are
        # destroyed with the same HP before then, no winner is inferable from
        # this minimal model.
        if self.time_s < self.duration_s:
            return None

        if not self.red.outpost_destroyed_ever and not self.blue.outpost_destroyed_ever:
            result = self._higher(self.red.outpost_hp, self.blue.outpost_hp)
            if result is not None:
                return result
        elif self.red.outpost_destroyed_ever != self.blue.outpost_destroyed_ever:
            return "blue" if self.red.outpost_destroyed_ever else "red"

        result = self._higher(self.red.attack_damage, self.blue.attack_damage)
        if result is not None:
            return result
        return self._higher(red_total_hp, blue_total_hp)

    def outcome(self, *, red_total_hp: float, blue_total_hp: float) -> str:
        if not self.is_terminal():
            return "running"
        winner = self.winner(red_total_hp=red_total_hp, blue_total_hp=blue_total_hp)
        return "draw" if winner is None else f"{winner}_win"

    @staticmethod
    def _higher(red_value: float, blue_value: float) -> Team | None:
        if red_value > blue_value:
            return "red"
        if blue_value > red_value:
            return "blue"
        return None

    @staticmethod
    def _validate_damage(amount: float) -> float:
        amount = float(amount)
        if amount < 0.0:
            raise ValueError("damage must be non-negative")
        return amount

    @staticmethod
    def _validate_sides(attacker: Team, defender: Team) -> None:
        if attacker == defender:
            raise ValueError("attacker and defender must differ")
        if attacker not in ("red", "blue") or defender not in ("red", "blue"):
            raise ValueError("team must be 'red' or 'blue'")
