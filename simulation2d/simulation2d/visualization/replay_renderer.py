"""Shared, ungridded battlefield renderer for every tactical replay."""
from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from ..world.environment import EFFECTIVE_DPS_ROLES, REFEREE_ROLE_NAMES, SENTRY_EFFECTIVE_DPS_ROLE, Unit
from ..world.semantic_map import SemanticMap


def battlefield_base(path: Path, width: int) -> Image.Image:
    image = Image.open(path).convert("RGBA")
    if width > 0 and image.width != width:
        height = max(1, round(image.height * width / image.width))
        resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
        image = image.resize((width, height), resample)
    return image


def building_hp(env: Any) -> dict[str, float]:
    return {
        "red_outpost": float(env.match.red.outpost_hp), "blue_outpost": float(env.match.blue.outpost_hp),
        "red_base": float(env.match.red.base_hp), "blue_base": float(env.match.blue.base_hp),
    }


def _world_to_px(cell: tuple[float, float], map_obj: SemanticMap, image: Image.Image) -> tuple[int, int]:
    x, y = cell
    return (round((x + 0.5) / map_obj.width * image.width),
            round((1.0 - (y + 0.5) / map_obj.height) * image.height))


def _range_ellipse(center: tuple[int, int], metres: float, map_obj: SemanticMap, image: Image.Image) -> tuple[int, int, int, int]:
    rx = metres / (map_obj.width * map_obj.resolution_m) * image.width
    ry = metres / (map_obj.height * map_obj.resolution_m) * image.height
    return (round(center[0] - rx), round(center[1] - ry), round(center[0] + rx), round(center[1] + ry))


def _role_label(unit: Unit) -> str:
    return {"hero": "Hero", "engineer": "Engineer", "infantry3": "infantry3", "infantry4": "infantry4",
            "aerial": "Areo", "sentry": "sentry"}.get(unit.role, unit.role)


def _target_for_command(unit: Unit, command: Any | None, units: list[Unit]) -> Unit | None:
    if command is None or not command.target_role:
        return None
    for candidate in units:
        if candidate.team != unit.team and candidate.alive and command.target_role in {candidate.role, REFEREE_ROLE_NAMES[candidate.role]}:
            return candidate
    return None


def _draw_range_envelopes(draw: ImageDraw.ImageDraw, map_obj: SemanticMap, image: Image.Image, units: list[Unit]) -> None:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    layer = ImageDraw.Draw(overlay)
    for unit in units:
        if not unit.alive or unit.weapon_damage <= 0.0:
            continue
        center = _world_to_px(unit.cell, map_obj, image)
        if unit.role in EFFECTIVE_DPS_ROLES or unit.role == SENTRY_EFFECTIVE_DPS_ROLE:
            layer.ellipse(_range_ellipse(center, 5.0, map_obj, image), outline=(255, 197, 69, 105), width=2)
            layer.ellipse(_range_ellipse(center, 3.0, map_obj, image), outline=(255, 83, 74, 145), width=2)
        elif unit.role == "hero":
            layer.ellipse(_range_ellipse(center, unit.attack_range, map_obj, image), outline=(171, 93, 255, 110), width=2)
    image.alpha_composite(overlay)


def _draw_sentry_goal(
    draw: ImageDraw.ImageDraw,
    map_obj: SemanticMap,
    image: Image.Image,
    units: list[Unit],
    sentry_state: dict[str, Any],
    font: ImageFont.ImageFont,
) -> None:
    goal_cell = sentry_state.get("goal_cell")
    if goal_cell is None:
        return
    goal = (int(goal_cell[0]), int(goal_cell[1]))
    goal_px = _world_to_px(goal, map_obj, image)
    sentry = next((unit for unit in units if unit.unit_id == 7 and unit.team == "red"), None)
    if sentry is not None:
        draw.line((_world_to_px(sentry.cell, map_obj, image), goal_px), fill=(255, 145, 25), width=2)
    radius = 7
    draw.ellipse(
        (goal_px[0] - radius, goal_px[1] - radius, goal_px[0] + radius, goal_px[1] + radius),
        fill=(255, 255, 255), outline=(255, 120, 0), width=3,
    )
    draw.line((goal_px[0] - 10, goal_px[1], goal_px[0] + 10, goal_px[1]), fill=(255, 120, 0), width=2)
    draw.line((goal_px[0], goal_px[1] - 10, goal_px[0], goal_px[1] + 10), fill=(255, 120, 0), width=2)
    goal_xy = sentry_state.get("goal_xy_m")
    if goal_xy is not None:
        label = f"goal ({float(goal_xy[0]):.2f}, {float(goal_xy[1]):.2f})m"
        draw.rectangle((goal_px[0] + 10, goal_px[1] - 14, goal_px[0] + 157, goal_px[1] + 1), fill=(255, 255, 255))
        draw.text((goal_px[0] + 13, goal_px[1] - 12), label, fill=(180, 72, 0), font=font)


def _draw_structures(draw: ImageDraw.ImageDraw, map_obj: SemanticMap, image: Image.Image, hp_state: dict[str, float], font: ImageFont.ImageFont) -> None:
    structures = (("R OP", map_obj.red_outpost, hp_state["red_outpost"], 1500.0, (220, 48, 48)),
                  ("B OP", map_obj.blue_outpost, hp_state["blue_outpost"], 1500.0, (30, 112, 225)),
                  ("R BASE", map_obj.red_base, hp_state["red_base"], 5000.0, (166, 34, 34)),
                  ("B BASE", map_obj.blue_base, hp_state["blue_base"], 5000.0, (20, 75, 174)))
    for label, cell, hp, maximum, color in structures:
        x, y = _world_to_px(cell, map_obj, image)
        box = (x - 38, y - 27, x + 38, y - 11)
        draw.rectangle(box, fill=(255, 255, 255), outline=color, width=2)
        width = max(1, round((box[2] - box[0] - 2) * max(0.0, min(1.0, hp / maximum))))
        draw.rectangle((box[0] + 1, box[1] + 1, box[0] + width, box[3] - 1), fill=color)
        draw.text((box[0] + 3, box[1] + 3), f"{label} {hp:.0f}", fill=(255, 255, 255), font=font)


def _draw_intents(draw: ImageDraw.ImageDraw, map_obj: SemanticMap, image: Image.Image, units: list[Unit], commands: dict[int, Any], events: tuple[dict[str, Any], ...], font: ImageFont.ImageFont) -> None:
    by_id = {unit.unit_id: unit for unit in units}
    for unit in units:
        if not unit.alive:
            continue
        command = commands.get(unit.unit_id)
        target = _target_for_command(unit, command, units)
        if target is not None:
            draw.line((_world_to_px(unit.cell, map_obj, image), _world_to_px(target.cell, map_obj, image)), fill=(80, 80, 80), width=1)
        elif command is not None and command.target_role in {"outpost", "前哨站", "base", "基地"}:
            defender = "blue" if unit.team == "red" else "red"
            structure = (map_obj.blue_outpost if defender == "blue" else map_obj.red_outpost) if command.target_role in {"outpost", "前哨站"} else (map_obj.blue_base if defender == "blue" else map_obj.red_base)
            draw.line((_world_to_px(unit.cell, map_obj, image), _world_to_px(structure, map_obj, image)), fill=(105, 68, 170), width=2)
    structures = {"red_outpost": map_obj.red_outpost, "blue_outpost": map_obj.blue_outpost, "red_base": map_obj.red_base, "blue_base": map_obj.blue_base}
    for event in events:
        attacker = by_id.get(int(event["attacker_id"]))
        if attacker is None:
            continue
        target = by_id.get(int(event["target_id"])) if event.get("target_id") is not None else None
        target_cell = target.cell if target is not None else structures.get(str(event["target_role"]))
        if target_cell is None:
            continue
        color = (255, 75, 65) if str(event["band"]) == "close_0_3m" else (255, 204, 65)
        draw.line((_world_to_px(attacker.cell, map_obj, image), _world_to_px(target_cell, map_obj, image)), fill=color, width=4)
        target_px = _world_to_px(target_cell, map_obj, image)
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
        top = center[1] - radius - 9
        draw.rectangle((center[0] - 13, top, center[0] + 13, top + 4), fill=(25, 25, 25))
        fill = round(26 * max(0.0, min(1.0, unit.hp / max(unit.max_hp, 1.0))))
        draw.rectangle((center[0] - 13, top, center[0] - 13 + fill, top + 4), fill=(92, 220, 106))
        draw.text((center[0] + radius + 2, center[1] - radius), f"{_role_label(unit)} {unit.hp:.0f}", fill=(20, 20, 20), font=font)


def frame(map_obj: SemanticMap, battlefield: Image.Image, hp_state: dict[str, float], units: list[Unit], commands: dict[int, Any], events: tuple[dict[str, Any], ...], sentry_state: dict[str, Any], step: int, profile: str, *, title: str, detail: str) -> Image.Image:
    image = battlefield.copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    _draw_range_envelopes(draw, map_obj, image, units)
    _draw_intents(draw, map_obj, image, units, commands, events, font)
    _draw_sentry_goal(draw, map_obj, image, units, sentry_state, font)
    _draw_structures(draw, map_obj, image, hp_state, font)
    _draw_units(draw, map_obj, image, units, font)
    draw.rectangle((8, 8, 790, 52), fill=(255, 255, 255), outline=(25, 25, 25), width=1)
    draw.text((13, 13), f"{title} | t={step:03d}s | {profile}", fill=(15, 23, 42), font=font)
    draw.text((13, 26), detail.format(hp=sentry_state["hp"]), fill=(15, 23, 42), font=font)
    goal_xy = sentry_state.get("goal_xy_m")
    goal_text = "goal=none" if goal_xy is None else f"goal=({float(goal_xy[0]):.2f}, {float(goal_xy[1]):.2f})m"
    target_text = str(sentry_state.get("target_label", "none"))
    fire_text = str(sentry_state.get("fire_label", "HOLD"))
    draw.text(
        (13, 39), f"published: {goal_text} | target={target_text} | fire={fire_text}",
        fill=(165, 70, 0), font=font,
    )
    return image.convert("RGB")


def write_mp4(images: list[Image.Image], path: Path, fps: float) -> None:
    if not images:
        return
    width, height = images[0].size
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(max(float(fps), 0.1)), "-i", "-", "-an", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
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
