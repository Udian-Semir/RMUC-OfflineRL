"""Replay a trained sentry policy in the interactive 2D world.

This is a diagnostic tool, not a trainer.  It runs one episode with a loaded
PPO checkpoint, records every non-learning unit's command/position/fire state,
and renders a lightweight trajectory preview without depending on matplotlib.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import yaml

from ..world.environment import SentryTacticalEnv, Unit
from ..policy.network import TacticalActorCritic
from ..resources import asset_path, config_path
from ..world.semantic_map import Cell, SemanticMap


DEFAULT_MAP_JSON = asset_path("semantic_map_aligned.json")
DEFAULT_OBSTACLE_MAP = asset_path("blackwhite_map.png")
DEFAULT_BATTLEFIELD_MAP = asset_path("semantic_map_aligned.png")


@dataclass
class UnitStats:
    team: str
    role: str
    unit_id: int
    start_cell: Cell
    end_cell: Cell
    moved_steps: int = 0
    manhattan_distance: int = 0
    shots_fired: int = 0
    hp_lost: float = 0.0


def _load_config(path: Path | None, checkpoint_payload: dict[str, Any]) -> dict[str, Any]:
    if path is not None:
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    # The PPO checkpoint intentionally records the training environment so its
    # policy shape/reward can be reproduced.  Sparring execution knobs are
    # simulator-side safety policy, though: old checkpoints do not know about
    # newly added legal-autoaim modes.  Fill only missing values from the live
    # offline-sparring config instead of silently replaying with legacy gates.
    with config_path("demo.yaml").open("r", encoding="utf-8") as handle:
        current_config = yaml.safe_load(handle)
    extra = checkpoint_payload.get("extra") or {}
    config = extra.get("config")
    if config:
        config = dict(config)
        env = dict(config.get("env") or {})
        for key in ("sparring_backend", "sparring_update_seconds", "sparring_device", "sparring_fire_policy"):
            if key not in env and key in current_config.get("env", {}):
                env[key] = current_config["env"][key]
        config["env"] = env
        return config
    return current_config


def _device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


def _tensor_obs(obs: dict[str, np.ndarray], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(obs["map"], device=device).unsqueeze(0),
        torch.as_tensor(obs["vector"], device=device).unsqueeze(0),
        torch.as_tensor(obs["goal_mask"], device=device).unsqueeze(0),
        torch.as_tensor(obs["target_mask"], device=device).unsqueeze(0),
    )


def _unit_name(unit: Unit) -> str:
    return f"{unit.team}_{unit.role}_{unit.unit_id}"


def _unit_snapshot(unit: Unit, prefix: str = "") -> dict[str, Any]:
    return {
        f"{prefix}cell_x": unit.cell[0],
        f"{prefix}cell_y": unit.cell[1],
        f"{prefix}hp": round(float(unit.hp), 3),
        f"{prefix}ammo": int(unit.ammo),
        f"{prefix}shots_fired": int(unit.shots_fired),
        f"{prefix}alive": bool(unit.alive),
    }


def _command_snapshot(command: Any | None) -> dict[str, Any]:
    if command is None:
        return {
            "command_goal_x": "",
            "command_goal_y": "",
            "command_fire_allowed": "",
            "command_target_role": "",
            "command_target_confidence": "",
        }
    return {
        "command_goal_x": command.goal_cell[0],
        "command_goal_y": command.goal_cell[1],
        "command_fire_allowed": bool(command.fire_allowed),
        "command_target_role": command.target_role or "",
        "command_target_confidence": round(float(command.target_confidence), 6),
    }


def _sentry_target_label(env: SentryTacticalEnv, target_idx: int) -> str:
    if 0 <= target_idx < env.ROBOT_TARGETS:
        target = env.enemies[target_idx]
        role = {
            "hero": "Hero",
            "engineer": "Engineer",
            "infantry3": "infantry3",
            "infantry4": "infantry4",
            "aerial": "Areo",
            "sentry": "sentry",
        }.get(target.role, target.role)
        return f"blue {role} #{target.unit_id}"
    if target_idx == env.BLUE_OUTPOST_TARGET:
        return "blue outpost"
    if target_idx == env.BLUE_BASE_TARGET:
        return "blue base"
    return "none"


def _published_sentry_state(env: SentryTacticalEnv, info: dict[str, Any]) -> dict[str, Any]:
    goal_cell = info.get("execution_goal_cell")
    goal_xy = info.get("execution_goal_xy_m")
    target_idx = int(info.get("executed_target_idx", env.NONE_TARGET))
    fire_mode = int(info.get("executed_fire_mode", env.FIRE_HOLD))
    return {
        "hp": float(env.sentry.hp),
        "goal_cell": tuple(goal_cell) if goal_cell is not None else None,
        "goal_xy_m": tuple(goal_xy) if goal_xy is not None else None,
        "target_idx": target_idx,
        "target_label": _sentry_target_label(env, target_idx),
        "fire_label": "ENGAGE" if fire_mode == env.FIRE_ENGAGE else "HOLD",
    }


def _cell_to_px(cell: Cell, height: int, scale: int) -> tuple[int, int]:
    x, y = cell
    return (x * scale + scale // 2, (height - 1 - y) * scale + scale // 2)


def _draw_base(map_obj: SemanticMap, scale: int) -> Image.Image:
    image = Image.new("RGB", (map_obj.width * scale, map_obj.height * scale), (246, 246, 246))
    draw = ImageDraw.Draw(image)
    for y in range(map_obj.height):
        for x in range(map_obj.width):
            if map_obj.hard_blocked[y, x]:
                y0 = (map_obj.height - 1 - y) * scale
                draw.rectangle((x * scale, y0, (x + 1) * scale - 1, y0 + scale - 1), fill=(28, 28, 28))
    for x in range(map_obj.width + 1):
        draw.line((x * scale, 0, x * scale, map_obj.height * scale), fill=(218, 218, 218))
    for y in range(map_obj.height + 1):
        draw.line((0, y * scale, map_obj.width * scale, y * scale), fill=(218, 218, 218))
    return image


def _draw_structures(draw: ImageDraw.ImageDraw, map_obj: SemanticMap, scale: int) -> None:
    specs = (
        ("R-OP", map_obj.red_outpost, (220, 50, 50)),
        ("B-OP", map_obj.blue_outpost, (45, 95, 220)),
        ("R-B", map_obj.red_base, (160, 40, 40)),
        ("B-B", map_obj.blue_base, (30, 70, 170)),
    )
    for label, cell, color in specs:
        cx, cy = _cell_to_px(cell, map_obj.height, scale)
        r = max(4, scale // 3)
        draw.rectangle((cx - r, cy - r, cx + r, cy + r), outline=color, width=2)
        draw.text((cx + r + 2, cy - r), label, fill=color)


def _draw_units(draw: ImageDraw.ImageDraw, map_obj: SemanticMap, units: list[Unit], scale: int) -> None:
    colors = {"red": (214, 48, 49), "blue": (9, 105, 218)}
    labels = {
        "hero": "Hero", "engineer": "Engineer", "infantry3": "infantry3",
        "infantry4": "infantry4", "aerial": "Areo", "sentry": "sentry",
    }
    for unit in units:
        if not unit.alive:
            continue
        cx, cy = _cell_to_px(unit.cell, map_obj.height, scale)
        radius = max(3, scale // 4)
        color = colors.get(unit.team, (50, 50, 50))
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color, outline=(255, 255, 255))
        draw.text((cx + radius + 2, cy - radius), labels.get(unit.role, unit.role), fill=color)


def _draw_combat_events(
    draw: ImageDraw.ImageDraw,
    map_obj: SemanticMap,
    units: list[Unit],
    events: tuple[dict[str, Any], ...],
    scale: int,
) -> None:
    """Draw only the range rings which actually produced damage this second."""
    by_id = {unit.unit_id: unit for unit in units}
    structures = {
        "red_outpost": map_obj.red_outpost,
        "blue_outpost": map_obj.blue_outpost,
        "red_base": map_obj.red_base,
        "blue_base": map_obj.blue_base,
    }
    for event in events:
        attacker = by_id.get(int(event["attacker_id"]))
        if attacker is None:
            continue
        attacker_px = _cell_to_px(attacker.cell, map_obj.height, scale)
        band = str(event["band"])
        if band in {"close_0_3m", "mid_3_5m"}:
            # The pale circle is the 5 m effective envelope; the red circle
            # is the 3 m high-damage envelope.  They are map metres because
            # one tactical cell equals one metre.
            for radius_m, color in ((5, (250, 190, 60)), (3, (236, 80, 55))):
                radius = int(radius_m * scale)
                draw.ellipse(
                    (attacker_px[0] - radius, attacker_px[1] - radius,
                     attacker_px[0] + radius, attacker_px[1] + radius),
                    outline=color,
                    width=1,
                )
        target_cell = structures.get(str(event["target_role"]))
        target_id = event.get("target_id")
        if target_id is not None and int(target_id) in by_id:
            target_cell = by_id[int(target_id)].cell
        if target_cell is None:
            continue
        target_px = _cell_to_px(target_cell, map_obj.height, scale)
        color = (236, 80, 55) if band == "close_0_3m" else (250, 190, 60)
        draw.line((attacker_px, target_px), fill=color, width=2)
        draw.text((target_px[0] + 2, target_px[1] + 2), f"-{float(event['damage']):.0f}", fill=color)


def _render_trajectory(map_obj: SemanticMap, trajectories: dict[str, list[Cell]], out_path: Path, scale: int = 28) -> None:
    image = _draw_base(map_obj, scale)
    draw = ImageDraw.Draw(image)
    _draw_structures(draw, map_obj, scale)
    palette = {
        "red": (214, 48, 49),
        "blue": (9, 105, 218),
    }
    for name, cells in trajectories.items():
        if len(cells) < 2:
            continue
        team = name.split("_", 1)[0]
        color = palette.get(team, (80, 80, 80))
        points = [_cell_to_px(cell, map_obj.height, scale) for cell in cells]
        draw.line(points, fill=color, width=2)
        sx, sy = points[0]
        ex, ey = points[-1]
        draw.ellipse((sx - 3, sy - 3, sx + 3, sy + 3), fill=(255, 255, 255), outline=color)
        draw.rectangle((ex - 4, ey - 4, ex + 4, ey + 4), fill=color)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def _render_gif(
    map_obj: SemanticMap,
    frames: list[tuple[int, list[Unit], tuple[dict[str, Any], ...]]],
    out_path: Path,
    *,
    scale: int = 24,
    stride: int = 3,
) -> None:
    images: list[Image.Image] = []
    font = ImageFont.load_default()
    for step, units, events in frames[::max(1, stride)]:
        image = _draw_base(map_obj, scale)
        draw = ImageDraw.Draw(image)
        _draw_structures(draw, map_obj, scale)
        _draw_units(draw, map_obj, units, scale)
        _draw_combat_events(draw, map_obj, units, events, scale)
        draw.rectangle((4, 4, 120, 20), fill=(255, 255, 255))
        draw.text((8, 7), f"t={step:03d}s", fill=(20, 20, 20), font=font)
        images.append(image)
    if not images:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(out_path, save_all=True, append_images=images[1:], duration=120, loop=0, optimize=True)


def _clone_units(units: list[Unit]) -> list[Unit]:
    return [Unit(**asdict(unit)) for unit in units]


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay one PPO episode in the interactive simulation2d world")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--map-json", type=Path, default=DEFAULT_MAP_JSON)
    parser.add_argument("--obstacle-map", type=Path, default=DEFAULT_OBSTACLE_MAP)
    parser.add_argument("--battlefield-map", type=Path, default=DEFAULT_BATTLEFIELD_MAP,
                        help="aligned ungridded battlefield background used by every replay renderer")
    parser.add_argument("--render-width", type=int, default=1120)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--sparring-backend",
        choices=("scripted", "offline"),
        default="offline",
        help="offline matches PPO training/replay; scripted is a self-contained debug opponent",
    )
    parser.add_argument("--sample", action="store_true", help="sample PPO actions instead of deterministic argmax")
    parser.add_argument(
        "--blue-profile",
        choices=("aggressive", "pressure", "measured"),
        default=None,
        help="force one episode-constant blue sparring profile for evaluation",
    )
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--gif-stride", type=int, default=2)
    parser.add_argument("--fps", type=float, default=8.0)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = _load_config(args.config, checkpoint)
    env_cfg = dict(config["env"])
    train_cfg = dict(config.get("train", {}))
    if args.seed is not None:
        env_cfg["seed"] = args.seed
    env_cfg["sparring_backend"] = args.sparring_backend
    if args.blue_profile is not None:
        env_cfg["blue_sparring_profiles"] = [args.blue_profile]
    if args.steps is not None:
        env_cfg["horizon"] = args.steps

    semantic_map = SemanticMap.from_aligned_json(args.map_json, obstacle_path=args.obstacle_map)
    env = SentryTacticalEnv(semantic_map=semantic_map, **env_cfg)
    device = torch.device(_device(args.device if args.device != "auto" else str(train_cfg.get("device", "auto"))))
    model = TacticalActorCritic(env.map_channels, env.vector_dim, env.n_goals, env.n_targets).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or Path("runs") / f"sentry_tactical_replay_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Keep PPO and database/hybrid replays visually identical without pulling
    # the database tool (and its pandas dependency) into this CLI.
    from ..visualization.replay_renderer import battlefield_base, building_hp, frame as render_replay_frame, write_mp4
    battlefield = battlefield_base(args.battlefield_map, args.render_width)

    obs = env.reset(seed=int(env_cfg.get("seed", 7)))
    unit_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    frames: list[tuple[int, list[Unit], dict[int, Any], tuple[dict[str, Any], ...], dict[str, float], dict[str, Any], str]] = []
    combat_rows: list[dict[str, Any]] = []
    trajectories: dict[str, list[Cell]] = {
        _unit_name(unit): [unit.cell]
        for unit in (env.sentry, *env.allies, *env.enemies)
    }
    start_units = {unit.unit_id: Unit(**asdict(unit)) for unit in (env.sentry, *env.allies, *env.enemies)}
    cumulative = {
        "reward": 0.0,
        "red_sentry_damage": 0.0,
        "red_outpost_damage": 0.0,
        "red_base_damage": 0.0,
        "blue_outpost_damage": 0.0,
        "blue_base_damage": 0.0,
        "damage_dealt_to_blue_robots": 0.0,
    }
    frames.append((
        0, _clone_units([env.sentry, *env.allies, *env.enemies]), {}, (),
        building_hp(env), {
            "hp": float(env.sentry.hp), "goal_cell": None, "goal_xy_m": None,
            "target_label": "none", "fire_label": "HOLD",
        }, env.blue_sparring_profile,
    ))

    done = False
    step = 0
    while not done:
        with torch.no_grad():
            action, _, _ = model.act(*_tensor_obs(obs, device), deterministic=not args.sample)
        action_np = action.squeeze(0).cpu().numpy()
        before = {unit.unit_id: Unit(**asdict(unit)) for unit in (env.sentry, *env.allies, *env.enemies)}
        obs, reward, done, info = env.step(action_np)
        step = int(env.step_count)
        units = [env.sentry, *env.allies, *env.enemies]
        commands = dict(env._sparring_commands)

        cumulative["reward"] += float(reward)
        cumulative["red_sentry_damage"] += float(info.get("damage_taken", 0.0))
        cumulative["red_outpost_damage"] += float(info.get("red_outpost_damage", 0.0))
        cumulative["red_base_damage"] += float(info.get("red_base_damage", 0.0))
        cumulative["blue_outpost_damage"] += float(info.get("blue_outpost_damage", 0.0))
        cumulative["blue_base_damage"] += float(info.get("blue_base_damage", 0.0))
        cumulative["damage_dealt_to_blue_robots"] += float(info.get("damage_dealt", 0.0))

        step_rows.append({
            "step": step,
            "time_s": round(float(info.get("match_time_s", step)), 3),
            "reward": round(float(reward), 6),
            "action_goal_idx": int(action_np[0]),
            "action_goal_name": info.get("goal_name", ""),
            "action_target_idx": int(action_np[1]),
            "action_fire_mode": int(action_np[2]),
            "executed_target_idx": int(info.get("executed_target_idx", env.NONE_TARGET)),
            "published_target": _sentry_target_label(
                env, int(info.get("executed_target_idx", env.NONE_TARGET))),
            "published_fire": (
                "ENGAGE" if int(info.get("executed_fire_mode", env.FIRE_HOLD)) == env.FIRE_ENGAGE else "HOLD"),
            "published_goal_x_m": (
                round(float(info["execution_goal_xy_m"][0]), 3) if info.get("execution_goal_xy_m") else ""),
            "published_goal_y_m": (
                round(float(info["execution_goal_xy_m"][1]), 3) if info.get("execution_goal_xy_m") else ""),
            "red_sentry_x": env.sentry.cell[0],
            "red_sentry_y": env.sentry.cell[1],
            "red_sentry_hp": round(float(env.sentry.hp), 3),
            "red_outpost_hp": round(float(info.get("red_outpost_hp", 0.0)), 3),
            "red_base_hp": round(float(info.get("red_base_hp", 0.0)), 3),
            "blue_outpost_hp": round(float(info.get("blue_outpost_hp", 0.0)), 3),
            "blue_base_hp": round(float(info.get("blue_base_hp", 0.0)), 3),
            "damage_taken": round(float(info.get("damage_taken", 0.0)), 3),
            "red_outpost_damage": round(float(info.get("red_outpost_damage", 0.0)), 3),
            "red_base_damage": round(float(info.get("red_base_damage", 0.0)), 3),
            "blue_outpost_damage": round(float(info.get("blue_outpost_damage", 0.0)), 3),
            "blue_base_damage": round(float(info.get("blue_base_damage", 0.0)), 3),
            "sparring_commands": int(info.get("sparring_commands", 0)),
            "outcome": info.get("outcome", ""),
        })

        for unit in units:
            name = _unit_name(unit)
            prev = before[unit.unit_id]
            row = {
                "step": step,
                "unit": name,
                "team": unit.team,
                "role": unit.role,
                "unit_id": unit.unit_id,
                **_unit_snapshot(prev, "before_"),
                **_unit_snapshot(unit, "after_"),
                "moved": unit.cell != prev.cell,
                "shots_delta": int(unit.shots_fired - prev.shots_fired),
                "hp_delta": round(float(unit.hp - prev.hp), 3),
                **_command_snapshot(commands.get(unit.unit_id)),
            }
            unit_rows.append(row)
            trajectories.setdefault(name, []).append(unit.cell)
        events = tuple(info.get("combat_events", ()))
        for event in events:
            combat_rows.append({"step": step, "time_s": round(float(info.get("match_time_s", step)), 3), **event})
        frames.append((
            step, _clone_units(units), commands, events, building_hp(env),
            _published_sentry_state(env, info), env.blue_sparring_profile,
        ))

    stats: dict[str, UnitStats] = {}
    for unit in (env.sentry, *env.allies, *env.enemies):
        start = start_units[unit.unit_id]
        cells = trajectories[_unit_name(unit)]
        moved_steps = sum(a != b for a, b in zip(cells, cells[1:]))
        manhattan = sum(abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in zip(cells, cells[1:]))
        stats[_unit_name(unit)] = UnitStats(
            team=unit.team,
            role=unit.role,
            unit_id=unit.unit_id,
            start_cell=start.cell,
            end_cell=unit.cell,
            moved_steps=int(moved_steps),
            manhattan_distance=int(manhattan),
            shots_fired=int(unit.shots_fired - start.shots_fired),
            hp_lost=float(start.hp - unit.hp),
        )

    with (out_dir / "steps.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(step_rows[0]))
        writer.writeheader()
        writer.writerows(step_rows)
    with (out_dir / "units.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(unit_rows[0]))
        writer.writeheader()
        writer.writerows(unit_rows)
    if combat_rows:
        with (out_dir / "combat_events.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(combat_rows[0]))
            writer.writeheader()
            writer.writerows(combat_rows)

    summary = {
        "checkpoint": str(args.checkpoint),
        "map_json": str(args.map_json),
        "steps": step,
        "deterministic": not args.sample,
        "blue_sparring_profile": env.blue_sparring_profile,
        "outcome": step_rows[-1].get("outcome", ""),
        "cumulative": {key: round(value, 3) for key, value in cumulative.items()},
        "unit_stats": {name: asdict(value) for name, value in stats.items()},
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    _render_trajectory(env.map, trajectories, out_dir / "trajectory.png")
    def render_frame(frame: tuple[int, list[Unit], dict[int, Any], tuple[dict[str, Any], ...], dict[str, float], dict[str, Any], str]) -> Image.Image:
        step, units, commands, events, hp_state, sentry_state, profile = frame
        return render_replay_frame(
            env.map, battlefield, hp_state, units, commands, events, sentry_state, step, profile,
            title="PPO: red sentry + 11 frozen offlineRL",
            detail=f"PPO sentry hp={sentry_state['hp']:.0f} | effective engagement rings: 3m / 5m",
        )

    if not args.no_gif:
        images = [render_frame(frame) for frame in frames[::max(1, args.gif_stride)]]
        images[0].save(
            out_dir / "replay.gif", save_all=True, append_images=images[1:],
            duration=round(1000 / max(args.fps, 1.0)), loop=0, optimize=True,
        )
    write_mp4([render_frame(frame) for frame in frames], out_dir / "replay.mp4", args.fps)

    blue_stats = [value for value in stats.values() if value.team == "blue"]
    print(f"replay_dir={out_dir}")
    print(f"steps={step} outcome={summary['outcome'] or 'not_terminal'} reward={cumulative['reward']:.3f}")
    print(
        "red_damage: sentry={:.1f} outpost={:.1f} base={:.1f}".format(
            cumulative["red_sentry_damage"],
            cumulative["red_outpost_damage"],
            cumulative["red_base_damage"],
        )
    )
    print(
        "blue_damage: robots={:.1f} outpost={:.1f} base={:.1f}".format(
            cumulative["damage_dealt_to_blue_robots"],
            cumulative["blue_outpost_damage"],
            cumulative["blue_base_damage"],
        )
    )
    print(
        "blue_sparring: moved_units={} total_move={} total_shots={}".format(
            sum(value.moved_steps > 0 for value in blue_stats),
            sum(value.manhattan_distance for value in blue_stats),
            sum(value.shots_fired for value in blue_stats),
        )
    )


if __name__ == "__main__":
    main()
