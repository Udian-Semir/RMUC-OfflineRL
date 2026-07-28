"""Exact observation adapter for frozen offline-RL sparring policies.

The policies in ``rm_runs`` were trained on a fixed 161-D referee observation,
not on :class:`SentryTacticalEnv`'s PPO observation.  This module makes that
translation explicit and reuses ``rm_rl.data.features.build_obs`` as the single
source of truth for feature ordering and normalisation semantics.

It intentionally stops at a high-level sparring command.  Feeding that command
into a particular robot simulator remains a separate movement/execution layer;
the command must still pass the same navigation and fire safety checks as any
scripted participant.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Literal

import numpy as np

from rm_rl.algos.action_spec import NO_TARGET
from rm_rl.data import schema as S
from rm_rl.data.features import Entity, GameArrays, build_obs, obs_dim
from rm_rl.deploy import MLPPolicyRunner

from .semantic_map import Cell, SemanticMap

if TYPE_CHECKING:
    from .env import SentryTacticalEnv, Unit


Team = Literal["red", "blue"]
_CAMP = {"red": S.CAMP_RED, "blue": S.CAMP_BLUE}


@dataclass(frozen=True)
class RefereeEntityState:
    """One mobile role in physical field coordinates, measured in metres."""

    role: str
    team: Team
    cell: Cell
    hp: float
    max_hp: float
    alive: bool = True
    heat17: float = 0.0
    heat17_max: float | None = None
    ammo17_fired: float = 0.0
    yaw_deg: float = 0.0
    power_w: float = 0.0
    vulnerable: bool = False

    @property
    def referee_role(self) -> str:
        return S.resolve_agent(self.role)


@dataclass(frozen=True)
class TacticalRefereeState:
    """Minimal complete snapshot needed by an offline-RL policy input."""

    time_s: float
    red_base_hp: float
    red_base_max_hp: float
    red_outpost_hp: float
    blue_base_hp: float
    blue_base_max_hp: float
    blue_outpost_hp: float
    entities: tuple[RefereeEntityState, ...]
    red_coin_left: float = 0.0
    red_coin_total: float = 0.0
    blue_coin_left: float = 0.0
    blue_coin_total: float = 0.0
    # The historical dataset uses these leak-safe fixed team priors.  A live
    # simulator does not have school histories, so their conservative default
    # is zero until a dedicated opponent-style estimator supplies them.
    team_features: tuple[float, float, float, float, float, float] = (0.0,) * 6


@dataclass(frozen=True)
class SparringCommand:
    """A frozen policy's high-level intent after de-normalisation."""

    goal_cell: Cell
    fire_allowed: bool
    target_role: str | None
    target_confidence: float
    target_probabilities: tuple[float, ...]


def from_tactical_env(env: "SentryTacticalEnv") -> TacticalRefereeState:
    """Create an adapter snapshot without implying missing roles exist.

    The current tactical demo only instantiates a subset of the six official
    mobile roles.  Absent roles stay absent (their 161-D slots become the same
    zero/alive=0 representation used by the offline data loader).
    """
    units: Iterable["Unit"] = (env.sentry, *env.allies, *env.enemies)
    entities = tuple(
        RefereeEntityState(
            role=unit.role,
            team=unit.team,
            cell=unit.cell,
            hp=unit.hp,
            max_hp=unit.max_hp,
            alive=unit.alive,
            heat17=unit.heat,
            heat17_max=260.0 if unit.role == "sentry" else None,
            ammo17_fired=max(0.0, 300.0 - unit.ammo) if unit.role == "sentry" else 0.0,
        )
        for unit in units
        if unit.role
    )
    return TacticalRefereeState(
        time_s=env.match.time_s,
        red_base_hp=env.match.red.base_hp,
        red_base_max_hp=env.match.red.base_max_hp,
        red_outpost_hp=env.match.red.outpost_hp,
        blue_base_hp=env.match.blue.base_hp,
        blue_base_max_hp=env.match.blue.base_max_hp,
        blue_outpost_hp=env.match.blue.outpost_hp,
        entities=entities,
    )


def build_offline_observation(
    state: TacticalRefereeState,
    *,
    ego_role: str,
    ego_team: Team,
) -> np.ndarray:
    """Return the exact raw 161-D vector expected by the frozen checkpoints."""
    ego_type = S.resolve_agent(ego_role)
    camp = _CAMP[ego_team]
    packed: dict[int, Entity] = {}
    roles_present: set[tuple[str, Team]] = set()
    for item in state.entities:
        role = item.referee_role
        if role not in S.MOBILE_TYPES:
            raise ValueError(f"unsupported mobile role: {item.role!r}")
        key = (role, item.team)
        if key in roles_present:
            raise ValueError(f"multiple entities for fixed referee slot: {key}")
        roles_present.add(key)
        packed[S.robot_id(role, _CAMP[item.team])] = _mobile_entity(item)
    if (ego_type, ego_team) not in roles_present:
        raise ValueError("offline policy ego role is absent from the tactical snapshot")
    if np.asarray(state.team_features, dtype=np.float32).shape != (6,):
        raise ValueError("team_features must contain exactly six values")

    packed[S.robot_id(S.TYPE_BASE, S.CAMP_RED)] = _building_entity(
        state.red_base_hp, state.red_base_max_hp, state.red_coin_left, state.red_coin_total)
    packed[S.robot_id(S.TYPE_OUTPOST, S.CAMP_RED)] = _building_entity(
        state.red_outpost_hp, S.MAX_HP[S.TYPE_OUTPOST], state.red_coin_left, state.red_coin_total)
    packed[S.robot_id(S.TYPE_BASE, S.CAMP_BLUE)] = _building_entity(
        state.blue_base_hp, state.blue_base_max_hp, state.blue_coin_left, state.blue_coin_total)
    packed[S.robot_id(S.TYPE_OUTPOST, S.CAMP_BLUE)] = _building_entity(
        state.blue_outpost_hp, S.MAX_HP[S.TYPE_OUTPOST], state.blue_coin_left, state.blue_coin_total)

    game = GameArrays(1, packed)
    obs = build_obs(
        game,
        camp=camp,
        agent_type=ego_type,
        t_max_ref=420.0,
        time_values=np.asarray([np.clip(state.time_s, 0.0, 420.0)], dtype=np.float32),
        team_feat=np.asarray(state.team_features, dtype=np.float32),
    )
    vector = obs[0]
    if vector.shape != (obs_dim(ego_type),):
        raise RuntimeError(f"offline observation shape {vector.shape} is not 161-D")
    if not np.isfinite(vector).all():
        raise ValueError("offline observation contains NaN/Inf")
    return vector


class OfflineSparringPolicy:
    """Run one frozen IQL/BC policy against an explicit tactical snapshot."""

    def __init__(self, run_dir: str, *, semantic_map: SemanticMap, team: Team, role: str, device: str = "cpu") -> None:
        self.semantic_map = semantic_map
        self.team = team
        self.role = role
        self.runner = MLPPolicyRunner(run_dir, device=device, camp=_CAMP[team])
        info = self.runner.info
        if info["action_mode"] != "tactical" or info["obs_dim"] != 161 or info["act_dim"] != 10:
            raise ValueError("sparring adapter requires a 161-D tactical offline checkpoint")

    def act(self, state: TacticalRefereeState) -> SparringCommand:
        obs = build_offline_observation(state, ego_role=self.role, ego_team=self.team)
        decoded = self.runner.step(obs)
        ego = _find_entity(state.entities, self.role, self.team)
        goal = self.semantic_map.nearest_free(self.semantic_map.clamp_cell((
            int(np.floor(ego.cell[0] + decoded["goal_dx"])),
            int(np.floor(ego.cell[1] + decoded["goal_dy"])),
        ))) or ego.cell
        target_index = decoded["target"]
        target_role = None if target_index is None or target_index == NO_TARGET else S.MOBILE_TYPES[int(target_index)]
        return SparringCommand(
            goal_cell=goal,
            fire_allowed=bool(decoded["fire"] > 0.5),
            target_role=target_role,
            target_confidence=float(decoded["target_conf"]),
            target_probabilities=tuple(float(value) for value in decoded["target_probs"]),
        )


def _mobile_entity(item: RefereeEntityState) -> Entity:
    heat_limit = _default_heat_limit(item.referee_role) if item.heat17_max is None else item.heat17_max
    return Entity(
        x=np.asarray([item.cell[0] + 0.5], dtype=np.float32),
        y=np.asarray([item.cell[1] + 0.5], dtype=np.float32),
        z=np.zeros(1, dtype=np.float32),
        hp=np.asarray([max(0.0, item.hp)], dtype=np.float32),
        maxhp=np.asarray([max(1.0, item.max_hp)], dtype=np.float32),
        yaw=np.asarray([item.yaw_deg], dtype=np.float32),
        power=np.asarray([max(0.0, item.power_w)], dtype=np.float32),
        heat17=np.asarray([max(0.0, item.heat17)], dtype=np.float32),
        heat17_max=np.asarray([max(0.0, heat_limit)], dtype=np.float32),
        heat42=np.zeros(1, dtype=np.float32),
        heat42_max=np.zeros(1, dtype=np.float32),
        ammo17=np.asarray([max(0.0, item.ammo17_fired)], dtype=np.float32),
        ammo42=np.zeros(1, dtype=np.float32),
        coin_left=np.zeros(1, dtype=np.float32),
        coin_total=np.zeros(1, dtype=np.float32),
        vuln=np.asarray([float(item.vulnerable)], dtype=np.float32),
        alive=np.asarray([float(item.alive and item.hp > 0.0)], dtype=np.float32),
    )


def _building_entity(hp: float, max_hp: float, coin_left: float, coin_total: float) -> Entity:
    values = np.asarray([max(0.0, hp)], dtype=np.float32)
    zeros = np.zeros(1, dtype=np.float32)
    return Entity(
        x=zeros, y=zeros, z=zeros,
        hp=values,
        maxhp=np.asarray([max(1.0, max_hp)], dtype=np.float32),
        yaw=np.full(1, S.YAW_SENTINEL, dtype=np.float32),
        power=zeros,
        heat17=zeros,
        heat17_max=zeros,
        heat42=zeros,
        heat42_max=zeros,
        ammo17=zeros,
        ammo42=zeros,
        coin_left=np.asarray([max(0.0, coin_left)], dtype=np.float32),
        coin_total=np.asarray([max(0.0, coin_total)], dtype=np.float32),
        vuln=zeros,
        alive=(values > 0.0).astype(np.float32),
    )


def _find_entity(entities: Iterable[RefereeEntityState], role: str, team: Team) -> RefereeEntityState:
    role_type = S.resolve_agent(role)
    for entity in entities:
        if entity.team == team and entity.referee_role == role_type:
            return entity
    raise ValueError("offline policy ego role is absent from the tactical snapshot")


def _default_heat_limit(referee_role: str) -> float:
    if referee_role in (S.TYPE_INFANTRY3, S.TYPE_INFANTRY4, S.TYPE_AERIAL, S.TYPE_SENTRY):
        return 260.0
    return 0.0
