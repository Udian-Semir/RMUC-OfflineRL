"""Visualise a recorded database sentry against 11 live offlineRL companions.

The recorded sentry supplies the red ego's state every second.  The other five
red units and six blue units are simulated by the frozen offlineRL sparring
policies.  This is a hybrid diagnostic, not a reconstruction of the original
database match.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sqlite3
import subprocess
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

from rm_rl.data import schema as S
from rm_rl.data.build_dataset import load_game_arrays
from ..world.environment import EFFECTIVE_DPS_ROLES, REFEREE_ROLE_NAMES, SentryTacticalEnv, Unit
from ..world.semantic_map import SemanticMap
from .replay_interactive import DEFAULT_MAP_JSON, DEFAULT_OBSTACLE_MAP
from ..resources import asset_path, config_path


DEFAULT_DB = Path("dataset/rmuc_2026_region_dataset.sqlite")
DEFAULT_CONFIG = config_path("online_ppo_aggressive_sparring.yaml")
DEFAULT_BATTLEFIELD_MAP = asset_path("semantic_map_aligned.png")


def _clone_units(units: list[Unit]) -> list[Unit]:
    return [Unit(**vars(unit)) for unit in units]


def _camp(name: str) -> str:
    table = {"red": S.CAMP_RED, "blue": S.CAMP_BLUE, "红": S.CAMP_RED, "蓝": S.CAMP_BLUE}
    try:
        return table[name.lower()]
    except KeyError as exc:
        raise ValueError("camp must be red or blue") from exc


def _first_game_id(db: Path) -> int:
    with sqlite3.connect(db) as con:
        row = con.execute("SELECT game_id FROM matches ORDER BY game_id LIMIT 1").fetchone()
    if row is None:
        raise RuntimeError(f"no games in {db}")
    return int(row[0])


def _recorded_sentry_rows(game, camp: str) -> list[dict[str, float]]:
    sentry = game.get(S.TYPE_SENTRY, camp)
    return [
        {
            "time_s": float(i + 1),
            "x_m": float(sentry.x[i]), "y_m": float(sentry.y[i]),
            "hp": float(sentry.hp[i]), "max_hp": float(sentry.maxhp[i]),
            "yaw_deg": float(sentry.yaw[i]), "heat": float(sentry.heat17[i]),
            "ammo17_fired": float(sentry.ammo17[i]), "power_w": float(sentry.power[i]),
        }
        for i in range(game.T)
    ]


def _battlefield_base(path: Path, width: int) -> Image.Image:
    """Load the drawn field itself; no tactical-grid lines are rendered."""
    image = Image.open(path).convert("RGBA")
    if width > 0 and image.width != width:
        height = max(1, round(image.height * width / image.width))
        # Pillow on the ROS/system Python is older than the Image.Resampling
        # enum; keep this renderer compatible with both Pillow generations.
        resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
        image = image.resize((width, height), resample)
    return image


def _world_to_px(cell: tuple[float, float], map_obj: SemanticMap, image: Image.Image) -> tuple[int, int]:
    # Cell positions are lower-left world coordinates.  The drawing PNG uses
    # conventional top-left pixels, hence the y inversion here only.
    x, y = cell
    return (
        round((x + 0.5) / map_obj.width * image.width),
        round((1.0 - (y + 0.5) / map_obj.height) * image.height),
    )


def _range_ellipse(center: tuple[int, int], metres: float, map_obj: SemanticMap, image: Image.Image) -> tuple[int, int, int, int]:
    rx = metres / (map_obj.width * map_obj.resolution_m) * image.width
    ry = metres / (map_obj.height * map_obj.resolution_m) * image.height
    return (round(center[0] - rx), round(center[1] - ry), round(center[0] + rx), round(center[1] + ry))


def _role_label(unit: Unit) -> str:
    labels = {
        "hero": "Hero", "engineer": "Engineer", "infantry3": "infantry3",
        "infantry4": "infantry4", "aerial": "Areo", "sentry": "sentry",
    }
    return labels.get(unit.role, unit.role)


def _target_for_command(unit: Unit, command: Any | None, units: list[Unit]) -> Unit | None:
    if command is None or not command.target_role:
        return None
    for candidate in units:
        if candidate.team == unit.team or not candidate.alive:
            continue
        if command.target_role in {candidate.role, REFEREE_ROLE_NAMES[candidate.role]}:
            return candidate
    return None


def _draw_range_envelopes(draw: ImageDraw.ImageDraw, map_obj: SemanticMap, image: Image.Image, units: list[Unit]) -> None:
    """Show every alive combat unit's effective attack envelope, not a grid."""
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    layer = ImageDraw.Draw(overlay)
    for unit in units:
        if not unit.alive or unit.weapon_damage <= 0.0:
            continue
        center = _world_to_px(unit.cell, map_obj, image)
        if unit.role in EFFECTIVE_DPS_ROLES:
            layer.ellipse(_range_ellipse(center, 5.0, map_obj, image), outline=(255, 197, 69, 105), width=2)
            layer.ellipse(_range_ellipse(center, 3.0, map_obj, image), outline=(255, 83, 74, 145), width=2)
        elif unit.role == "hero":
            layer.ellipse(_range_ellipse(center, unit.attack_range, map_obj, image), outline=(171, 93, 255, 110), width=2)
        elif unit.role == "sentry":
            layer.ellipse(_range_ellipse(center, unit.attack_range, map_obj, image), outline=(47, 201, 220, 105), width=2)
    image.alpha_composite(overlay)


def _building_hp(env: SentryTacticalEnv) -> dict[str, float]:
    return {
        "red_outpost": float(env.match.red.outpost_hp), "blue_outpost": float(env.match.blue.outpost_hp),
        "red_base": float(env.match.red.base_hp), "blue_base": float(env.match.blue.base_hp),
    }


def _draw_structure_status(
    draw: ImageDraw.ImageDraw,
    map_obj: SemanticMap,
    image: Image.Image,
    hp_state: dict[str, float],
    font: ImageFont.ImageFont,
) -> None:
    structures = (
        ("R OP", map_obj.red_outpost, hp_state["red_outpost"], 1500.0, (220, 48, 48)),
        ("B OP", map_obj.blue_outpost, hp_state["blue_outpost"], 1500.0, (30, 112, 225)),
        ("R BASE", map_obj.red_base, hp_state["red_base"], 5000.0, (166, 34, 34)),
        ("B BASE", map_obj.blue_base, hp_state["blue_base"], 5000.0, (20, 75, 174)),
    )
    for label, cell, hp, maximum, color in structures:
        x, y = _world_to_px(cell, map_obj, image)
        box = (x - 38, y - 27, x + 38, y - 11)
        draw.rectangle(box, fill=(255, 255, 255), outline=color, width=2)
        width = max(1, round((box[2] - box[0] - 2) * max(0.0, min(1.0, hp / maximum))))
        draw.rectangle((box[0] + 1, box[1] + 1, box[0] + width, box[3] - 1), fill=color)
        draw.text((box[0] + 3, box[1] + 3), f"{label} {hp:.0f}", fill=(255, 255, 255), font=font)


def _draw_intents_and_attacks(
    draw: ImageDraw.ImageDraw,
    map_obj: SemanticMap,
    image: Image.Image,
    units: list[Unit],
    commands: dict[int, Any],
    events: tuple[dict[str, Any], ...],
    font: ImageFont.ImageFont,
) -> None:
    by_id = {unit.unit_id: unit for unit in units}
    # Thin gray lines are current learned vehicle target intentions.  A bright
    # line and a damage number appear only when the combat state machine really
    # applied HP damage in this second.
    for unit in units:
        if not unit.alive:
            continue
        command = commands.get(unit.unit_id)
        target = _target_for_command(unit, command, units)
        if target is not None:
            draw.line((_world_to_px(unit.cell, map_obj, image), _world_to_px(target.cell, map_obj, image)), fill=(80, 80, 80), width=1)
        elif command is not None and command.target_role in {"outpost", "前哨站", "base", "基地"}:
            defender = "blue" if unit.team == "red" else "red"
            if command.target_role in {"outpost", "前哨站"}:
                structure = map_obj.blue_outpost if defender == "blue" else map_obj.red_outpost
            else:
                structure = map_obj.blue_base if defender == "blue" else map_obj.red_base
            draw.line((_world_to_px(unit.cell, map_obj, image), _world_to_px(structure, map_obj, image)), fill=(105, 68, 170), width=2)

    structures = {
        "red_outpost": map_obj.red_outpost, "blue_outpost": map_obj.blue_outpost,
        "red_base": map_obj.red_base, "blue_base": map_obj.blue_base,
    }
    for event in events:
        attacker = by_id.get(int(event["attacker_id"]))
        if attacker is None:
            continue
        target = by_id.get(int(event["target_id"])) if event.get("target_id") is not None else None
        target_cell = target.cell if target is not None else structures.get(str(event["target_role"]))
        if target_cell is None:
            continue
        source_px = _world_to_px(attacker.cell, map_obj, image)
        target_px = _world_to_px(target_cell, map_obj, image)
        band = str(event["band"])
        color = (255, 75, 65) if band == "close_0_3m" else (255, 204, 65)
        draw.line((source_px, target_px), fill=color, width=4)
        draw.text((target_px[0] + 4, target_px[1] + 3), f"-{float(event['damage']):.0f}", fill=color, font=font)


def _draw_units(draw: ImageDraw.ImageDraw, map_obj: SemanticMap, image: Image.Image, units: list[Unit], font: ImageFont.ImageFont) -> None:
    colors = {"red": (224, 54, 54), "blue": (35, 114, 228)}
    for unit in units:
        center = _world_to_px(unit.cell, map_obj, image)
        color = colors[unit.team] if unit.alive else (80, 80, 80)
        radius = 10 if unit.alive else 7
        draw.ellipse((center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius), fill=color, outline=(255, 255, 255), width=2)
        if unit.unit_id == 7:
            draw.ellipse((center[0] - radius - 3, center[1] - radius - 3, center[0] + radius + 3, center[1] + radius + 3), outline=(255, 164, 40), width=3)
        hp_width = 26
        top = center[1] - radius - 9
        draw.rectangle((center[0] - hp_width // 2, top, center[0] + hp_width // 2, top + 4), fill=(25, 25, 25))
        fill = round(hp_width * max(0.0, min(1.0, unit.hp / max(unit.max_hp, 1.0))))
        draw.rectangle((center[0] - hp_width // 2, top, center[0] - hp_width // 2 + fill, top + 4), fill=(92, 220, 106))
        draw.text((center[0] + radius + 2, center[1] - radius), f"{_role_label(unit)} {unit.hp:.0f}", fill=(20, 20, 20), font=font)


def _frame(
    map_obj: SemanticMap,
    battlefield: Image.Image,
    hp_state: dict[str, float],
    units: list[Unit],
    commands: dict[int, Any],
    events: tuple[dict[str, Any], ...],
    recorded: dict[str, float],
    step: int,
    profile: str,
    title: str | None = None,
    detail: str | None = None,
) -> Image.Image:
    image = battlefield.copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    _draw_range_envelopes(draw, map_obj, image, units)
    _draw_intents_and_attacks(draw, map_obj, image, units, commands, events, font)
    _draw_structure_status(draw, map_obj, image, hp_state, font)
    _draw_units(draw, map_obj, image, units, font)
    draw.rectangle((8, 8, 570, 39), fill=(255, 255, 255), outline=(25, 25, 25), width=1)
    title = title or "HYBRID: recorded red sentry + 11 frozen offlineRL"
    detail = detail or f"REC sentry hp={recorded['hp']:.0f} | rings: vehicle 3m/5m, hero 6.5m, sentry 8m"
    draw.text((13, 13), f"{title} | t={step:03d}s | {profile}", fill=(15, 23, 42), font=font)
    draw.text((13, 26), detail, fill=(15, 23, 42), font=font)
    return image.convert("RGB")


def _write_mp4(images: list[Image.Image], path: Path, fps: float) -> None:
    """Encode every simulation frame once; GIF optimisation may discard them."""
    if not images:
        return
    width, height = images[0].size
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(max(float(fps), 0.1)),
        "-i", "-", "-an", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(path),
    ]
    try:
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required to render replay.mp4") from exc
    assert process.stdin is not None
    try:
        for image in images:
            process.stdin.write(image.convert("RGB").tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg could not encode replay.mp4")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--game-id", type=int, default=None)
    parser.add_argument("--camp", default="red")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--map-json", type=Path, default=DEFAULT_MAP_JSON)
    parser.add_argument("--obstacle-map", type=Path, default=DEFAULT_OBSTACLE_MAP)
    parser.add_argument("--battlefield-map", type=Path, default=DEFAULT_BATTLEFIELD_MAP,
                        help="aligned drawn battlefield PNG used as the ungridded render background")
    parser.add_argument("--render-width", type=int, default=1120,
                        help="render width in pixels; 0 preserves the source battlefield size")
    parser.add_argument("--profile", choices=("aggressive", "pressure", "measured"), default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--max-seconds", type=int, default=None)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--gif-stride", type=int, default=2)
    parser.add_argument("--no-video", action="store_true",
                        help="skip the full per-second MP4 export")
    parser.add_argument("--live", action="store_true", help="show an OpenCV real-time window; Esc stops playback")
    args = parser.parse_args()

    camp = _camp(args.camp)
    game_id = args.game_id if args.game_id is not None else _first_game_id(args.db)
    with sqlite3.connect(args.db) as con:
        game = load_game_arrays(con, int(game_id))
        match = con.execute('SELECT "红方学校", "蓝方学校", "胜方", "时长秒" FROM matches WHERE game_id=?', (int(game_id),)).fetchone()
    if game.T == 0:
        raise RuntimeError(f"game_id={game_id} has no timeseries")
    recorded = _recorded_sentry_rows(game, camp)
    if args.max_seconds is not None:
        recorded = recorded[:max(1, args.max_seconds)]

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    env_cfg = dict(config["env"])
    env_cfg["sparring_backend"] = "offline"
    if args.profile:
        env_cfg["blue_sparring_profiles"] = [args.profile]
    map_obj = SemanticMap.from_aligned_json(args.map_json, obstacle_path=args.obstacle_map)
    env = SentryTacticalEnv(semantic_map=map_obj, **env_cfg)
    env.reset(seed=int(env_cfg.get("seed", 31)))
    battlefield = _battlefield_base(args.battlefield_map, args.render_width)

    out_dir = args.out_dir or Path("runs") / f"database_sentry_sparring_{game_id}_{camp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    replay_rows: list[dict[str, object]] = []
    unit_rows: list[dict[str, object]] = []
    combat_rows: list[dict[str, object]] = []
    frames: list[tuple[int, list[Unit], dict[int, Any], tuple[dict[str, Any], ...], dict[str, float], dict[str, float], str]] = []
    live = bool(args.live)
    for step, state in enumerate(recorded, start=1):
        env.set_external_sentry_state(**state)
        _, reward, done, info = env.step((0, env.NONE_TARGET, env.FIRE_HOLD))
        # Restore the database value after the companion step so the visual is
        # honest about the recorded sentry, not the simulated damage outcome.
        env.set_external_sentry_state(**state)
        units = [env.sentry, *env.allies, *env.enemies]
        commands = dict(env._sparring_commands)
        events = tuple(info.get("combat_events", ()))
        hp_state = _building_hp(env)
        frame = _frame(env.map, battlefield, hp_state, units, commands, events, state, step, env.blue_sparring_profile)
        frames.append((step, _clone_units(units), commands, events, hp_state, dict(state), env.blue_sparring_profile))
        replay_rows.append({
            "step": step, "game_id": int(game_id), "camp": camp,
            "recorded_x_m": state["x_m"], "recorded_y_m": state["y_m"],
            "recorded_hp": state["hp"], "recorded_yaw_deg": state["yaw_deg"],
            "profile": env.blue_sparring_profile, "reward": float(reward),
            "red_sentry_damage": float(info.get("damage_taken", 0.0)),
            "red_outpost_damage": float(info.get("red_outpost_damage", 0.0)),
            "red_base_damage": float(info.get("red_base_damage", 0.0)),
            "blue_outpost_damage": float(info.get("blue_outpost_damage", 0.0)),
            "blue_base_damage": float(info.get("blue_base_damage", 0.0)),
            "red_outpost_hp": hp_state["red_outpost"],
            "blue_outpost_hp": hp_state["blue_outpost"],
            "red_base_hp": hp_state["red_base"],
            "blue_base_hp": hp_state["blue_base"],
        })
        for unit in units:
            command = commands.get(unit.unit_id)
            regions = env.map.region_ids_at(unit.cell)
            unit_rows.append({
                "step": step, "team": unit.team, "role": unit.role, "unit_id": unit.unit_id,
                "cell_x": unit.cell[0], "cell_y": unit.cell[1], "hp": round(float(unit.hp), 3),
                "max_hp": round(float(unit.max_hp), 3), "alive": bool(unit.alive),
                "goal_x": command.goal_cell[0] if command else "",
                "goal_y": command.goal_cell[1] if command else "",
                "fire_gate": bool(command.fire_allowed) if command else "",
                "target_role": command.target_role if command and command.target_role else "",
                "target_confidence": round(float(command.target_confidence), 6) if command else "",
                "semantic_regions": ";".join(regions),
                "healing_active": any(env.map.region_kinds[region] == "healing" for region in regions),
                "defense_pct": round(100.0 * env._unit_defense_bonus(unit), 1),
                "cooling_per_second": round(env._heat_cooling_rate(unit), 1),
                "tunnel_defense_remaining_s": round(max(0.0, unit.tunnel_defense_until_s - env.match.time_s), 1),
                "tunnel_cooling_remaining_s": round(max(0.0, unit.tunnel_cooling_until_s - env.match.time_s), 1),
            })
        for event in events:
            combat_rows.append({"step": step, "time_s": state["time_s"], **event})
        if live:
            try:
                import cv2
                cv2.imshow("RMUC database sentry + offlineRL sparring", np.asarray(frame)[:, :, ::-1])
                if cv2.waitKey(max(1, int(1000 / max(args.fps, 1.0)))) & 0xFF == 27:
                    break
            except Exception as exc:
                print(f"live display unavailable: {exc}")
                live = False
        if done:
            break
    if args.live:
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass

    with (out_dir / "replay.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(replay_rows[0]))
        writer.writeheader()
        writer.writerows(replay_rows)
    with (out_dir / "units.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(unit_rows[0]))
        writer.writeheader()
        writer.writerows(unit_rows)
    if combat_rows:
        with (out_dir / "combat_events.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(combat_rows[0]))
            writer.writeheader()
            writer.writerows(combat_rows)
    images = [
        _frame(map_obj, battlefield, hp_state, units, commands, events, state, step, profile)
        for step, units, commands, events, hp_state, state, profile in frames[::max(1, args.gif_stride)]
    ]
    images[0].save(out_dir / "replay.gif", save_all=True, append_images=images[1:], duration=round(1000 / max(args.fps, 1.0)), loop=0, optimize=True)
    if not args.no_video:
        _write_mp4([
            _frame(map_obj, battlefield, hp_state, units, commands, events, state, step, profile)
            for step, units, commands, events, hp_state, state, profile in frames
        ], out_dir / "replay.mp4", args.fps)
    images[-1].save(out_dir / "final.png")
    cumulative = {key: round(float(sum(row[key] for row in replay_rows)), 3) for key in (
        "reward", "red_sentry_damage", "red_outpost_damage", "red_base_damage", "blue_outpost_damage", "blue_base_damage")}
    summary = {
        "game_id": int(game_id), "camp": camp, "match": match,
        "recorded_sentry_seconds_available": len(recorded),
        "hybrid_seconds_played": len(replay_rows), "sparring_units": 11,
        "simulated_outcome": info.get("outcome", ""),
        "profile": env.blue_sparring_profile,
        "recorded_sentry_source": "official sqlite timeseries",
        "companions_source": "frozen offlineRL plus tactical referee",
        "battlefield_map": str(args.battlefield_map),
        "render_contract": "ungridded drawn battlefield; rings/intent lines/attack lines are tactical simulation overlays",
        "cumulative": cumulative,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"game_id={game_id} camp={camp} hybrid_seconds={len(replay_rows)}/{len(recorded)}")
    print(f"out_dir={out_dir}")
    print(json.dumps(cumulative, ensure_ascii=False))


if __name__ == "__main__":
    main()
