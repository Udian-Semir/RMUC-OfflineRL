"""Render a complete held-out offlineRL replay with IQL policy overlays.

The replay is a behaviour demonstration: every displayed game state comes from
one held-out official match, and the orange overlay is the trained IQL policy's
recommendation at that exact second.  It is intentionally useful without the
future interactive tactical environment.

Example:
    python -m rm_rl.eval.render_offline_replay \
        --data data/infantry_tactical \
        --run rm_runs/infantry_iql_tactical \
        --out rm_runs/sparring_visuals/offline_iql_replay.mp4
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-rmrl")

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.lines import Line2D
import numpy as np
import torch

from ..algos.action_spec import NO_TARGET
from ..data import schema as S
from ..data.features import obs_feature_names
from ..deploy import load_policy
from .plot_sparring_scenarios import SHORT_NAME, _ego_xy, _entities, _index


# Registered against the full 1454 x 804 occupancy-candidate image.  The user
# identified the edge structures as bases and the two small black structures on
# opposite sides of the central highland as outposts.  This is a display
# registration only; final planning geometry remains pending semantic-map data.
RMUC2026_LANDMARKS = {
    "red_base": (2.73, 7.45),
    "blue_base": (25.32, 7.45),
    "red_outpost": (10.63, 3.86),
    "blue_outpost": (16.95, 11.12),
}


def _policy_actions(model, norm, obs: np.ndarray, device: str, batch: int = 8192):
    out = []
    with torch.no_grad():
        for start in range(0, len(obs), batch):
            x = torch.as_tensor(obs[start:start + batch], dtype=torch.float32,
                                device=device)
            out.append(model.act(norm.normalize(x), deterministic=True).cpu().numpy())
    return np.concatenate(out, axis=0)


def _choose_episode(z, ix: dict[str, int]):
    """Select a held-out episode that visibly contains both combat and stakes."""
    obs, act, valid = z["obs"], z["act"], z["valid"] > 0.5
    scores = []
    for ep, (start, length) in enumerate(zip(z["ep_start"], z["ep_len"])):
        sl = slice(int(start), int(start + length))
        live = valid[sl]
        fire = ((act[sl, 2] > 0.5) & live).sum()
        named = ((act[sl, 3:].argmax(axis=1) != NO_TARGET) & live).sum()
        b = obs[sl]
        change = 0.0
        for name in ("build.own.base_hp", "build.enemy.base_hp",
                     "build.own.outpost_hp", "build.enemy.outpost_hp"):
            change += float(np.abs(np.diff(b[:, ix[name]])).sum())
        # One base-HP percent is more interesting to watch than a few idle
        # seconds, but active target selection remains the dominant signal.
        scores.append(float(fire * 2 + named + change * 1500.0))
    return int(np.argmax(scores))


def _draw_arena(ax):
    ax.set_facecolor("#f8fafc")
    ax.add_patch(plt.Rectangle((0, 0), S.FIELD_X, S.FIELD_Y, fill=False,
                               linewidth=2.0, edgecolor="#243447"))
    ax.axvline(S.FIELD_X / 2, color="#94a3b8", linewidth=1.0, linestyle="--")
    ax.axvspan(0, S.FIELD_X / 2, color="#fee2e2", alpha=0.35, zorder=0)
    ax.axvspan(S.FIELD_X / 2, S.FIELD_X, color="#dbeafe", alpha=0.35, zorder=0)
    for x, name, own in ((1.2, "own base", True), (7.0, "own outpost", True),
                         (21.0, "enemy outpost", False), (26.8, "enemy base", False)):
        color = "#991b1b" if own else "#1d4ed8"
        ax.scatter([x], [S.FIELD_Y / 2], marker="s", s=75, color=color, zorder=2)
        ax.text(x, S.FIELD_Y / 2 + 0.55, name, ha="center", va="bottom",
                fontsize=7, color=color)
    ax.set_xlim(-0.5, S.FIELD_X + 0.5)
    ax.set_ylim(-0.5, S.FIELD_Y + 0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([0, 7, 14, 21, 28])
    ax.set_yticks([0, 7.5, 15])
    ax.tick_params(labelsize=8)
    ax.set_xlabel("canonical field x (m): own half -> enemy half", fontsize=8)


def _target_name(action: np.ndarray) -> tuple[str, int]:
    target = int(np.argmax(action[3:]))
    return ("none" if target == NO_TARGET else SHORT_NAME[S.MOBILE_TYPES[target]],
            target)


def _actual_xy(x: float, y: float, camp: int) -> tuple[float, float]:
    """Undo blue-side canonicalisation for display on the real red/blue map."""
    return (S.FIELD_X - x, S.FIELD_Y - y) if camp == 1 else (x, y)


def _camp_name(camp: int) -> str:
    return "BLUE" if camp == 1 else "RED"


def _draw_state(ax, raw: np.ndarray, action: np.ndarray, scale: np.ndarray,
                ix: dict[str, int], episode_time: int, game_id: int, *, compact=False):
    _draw_arena(ax)
    ex, ey = _ego_xy(raw, ix)
    allies = _entities(raw, ix, "ally")
    enemies = _entities(raw, ix, "enemy")
    for rtype, x, y, hp in allies:
        if abs(x - ex) < 1e-3 and abs(y - ey) < 1e-3:
            continue
        ax.scatter([x], [y], s=46, color="#dc2626", zorder=4)
        if not compact:
            ax.text(x + 0.20, y + 0.20, f"{SHORT_NAME[rtype]} {hp:.0%}",
                    fontsize=7, color="#7f1d1d")
    for rtype, x, y, hp in enemies:
        ax.scatter([x], [y], s=52, marker="X", color="#2563eb", zorder=4)
        if not compact:
            ax.text(x + 0.20, y + 0.20, f"{SHORT_NAME[rtype]} {hp:.0%}",
                    fontsize=7, color="#1e3a8a")
    ax.scatter([ex], [ey], s=220, marker="*", color="#b91c1c", edgecolor="white",
               linewidth=0.8, zorder=6)
    ax.text(ex + 0.25, ey - 0.52, "offlineRL ego", fontsize=7,
            color="#7f1d1d", weight="bold")

    dx, dy = action[:2] * scale[:2]
    ax.arrow(ex, ey, dx, dy, width=0.055, head_width=0.45, head_length=0.55,
             length_includes_head=True, color="#d97706", alpha=0.95, zorder=7)
    target_text, target = _target_name(action)
    if target != NO_TARGET:
        wanted = S.MOBILE_TYPES[target]
        for rtype, tx, ty, _ in enemies:
            if rtype == wanted:
                ax.plot([ex, tx], [ey, ty], linestyle=(0, (3, 3)), color="#d97706",
                        linewidth=1.4, zorder=3)
                break

    fire = "ON" if action[2] > 0.5 else "off"
    status = (f"IQL action: goal ({dx:+.1f}, {dy:+.1f}) m | fire {fire} | "
              f"target {target_text}")
    hp = lambda name: float(raw[ix[name]])
    buildings = (f"own base {hp('build.own.base_hp'):.0%}, "
                 f"enemy base {hp('build.enemy.base_hp'):.0%}, "
                 f"own outpost {hp('build.own.outpost_hp'):.0%}, "
                 f"enemy outpost {hp('build.enemy.outpost_hp'):.0%}")
    ax.set_title(f"Held-out match {game_id} | t={episode_time:03d}s", fontsize=11, loc="left")
    if not compact:
        ax.text(0.0, -0.20, buildings + "\n" + status, transform=ax.transAxes,
                fontsize=8.5, va="top", family="monospace")


def _draw_hp(ax, raw_episode: np.ndarray, ix: dict[str, int], current: int):
    ax.clear()
    t = np.arange(1, len(raw_episode) + 1)
    lines = (("build.own.base_hp", "own base", "#b91c1c"),
             ("build.enemy.base_hp", "enemy base", "#2563eb"),
             ("build.own.outpost_hp", "own outpost", "#ef4444"),
             ("build.enemy.outpost_hp", "enemy outpost", "#60a5fa"))
    for key, label, color in lines:
        ax.plot(t, raw_episode[:, ix[key]], label=label, color=color, linewidth=1.7)
    ax.axvline(current + 1, color="#d97706", linewidth=1.5, linestyle="--")
    ax.set_xlim(1, len(raw_episode))
    ax.set_ylim(-0.03, 1.05)
    ax.set_ylabel("recorded HP fraction", fontsize=8)
    ax.set_xlabel("official match time (s)", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.2)
    ax.legend(ncols=2, fontsize=7, loc="lower left")


def _save_storyboard(path: str, raw: np.ndarray, pred: np.ndarray, scale: np.ndarray,
                     ix: dict[str, int], game_id: int):
    frame_ids = np.linspace(0, len(raw) - 1, 12, dtype=int)
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), constrained_layout=True)
    for ax, frame in zip(axes.flat, frame_ids):
        _draw_state(ax, raw[frame], pred[frame], scale, ix, frame + 1, game_id,
                    compact=True)
    fig.legend(handles=[
        Line2D([0], [0], color="#d97706", lw=2, label="IQL 5 s navigation recommendation"),
        Line2D([0], [0], color="#d97706", lw=1.5, linestyle=(0, (3, 3)), label="IQL selected target"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#b91c1c", markersize=10, label="controlled unit"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor="#2563eb", markersize=8, label="known enemy"),
    ], ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.03), fontsize=8)
    fig.suptitle("OfflineRL IQL replay storyboard: held-out official match states", fontsize=13, y=1.08)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _render_continuous_opencv(path: str, raw: np.ndarray, pred: np.ndarray,
                              scale: np.ndarray, ix: dict[str, int], game_id: int,
                              camp: int, fps: int, stride: int,
                              map_image: str | None = None):
    """Fast, one-recorded-second-per-frame MP4 renderer.

    Matplotlib is convenient for a storyboard but clearing two axes hundreds of
    times can exceed an interactive command's time limit.  This renderer keeps
    the same information and draws directly into video frames.
    """
    import cv2

    width, height = 1280, 860
    left, top, field_w, field_h = 45, 135, 1190, 658
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (width, height))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open the requested MP4 output")

    def point(x, y):
        return (int(left + x / S.FIELD_X * field_w),
                int(top + (S.FIELD_Y - y) / S.FIELD_Y * field_h))

    map_canvas = None
    if map_image:
        map_canvas = cv2.imread(map_image, cv2.IMREAD_COLOR)
        if map_canvas is None:
            raise FileNotFoundError(f"could not read --map-image {map_image}")
        map_canvas = cv2.resize(map_canvas, (field_w, field_h), interpolation=cv2.INTER_AREA)

    def dashed(canvas, a, b, color, step=12):
        ax, ay = a
        bx, by = b
        dist = max(int(np.hypot(bx - ax, by - ay)), 1)
        for start in range(0, dist, step * 2):
            u0, u1 = start / dist, min(start + step, dist) / dist
            p0 = (int(ax + (bx - ax) * u0), int(ay + (by - ay) * u0))
            p1 = (int(ax + (bx - ax) * u1), int(ay + (by - ay) * u1))
            cv2.line(canvas, p0, p1, color, 2, cv2.LINE_AA)

    frame_ids = np.arange(0, len(raw), max(1, stride), dtype=int)
    for frame in frame_ids:
        image = np.full((height, width, 3), 248, dtype=np.uint8)
        # OpenCV uses BGR.  The supplied occupancy candidate is the visual
        # field layer; the legacy halves remain available when no map is passed.
        if map_canvas is not None:
            image[top:top + field_h, left:left + field_w] = map_canvas
        else:
            cv2.rectangle(image, (left, top), (left + field_w // 2, top + field_h),
                          (235, 240, 255), -1)
            cv2.rectangle(image, (left + field_w // 2, top), (left + field_w, top + field_h),
                          (255, 242, 230), -1)
        cv2.rectangle(image, (left, top), (left + field_w, top + field_h),
                      (52, 65, 85), 2)

        state, action = raw[frame], pred[frame]
        ex, ey = _ego_xy(state, ix)
        ex, ey = _actual_xy(ex, ey, camp)
        ego_point = point(ex, ey)
        enemies = _entities(state, ix, "enemy")
        own_label, enemy_label = _camp_name(camp), _camp_name(1 - camp)
        own_color = (220, 105, 35) if camp == 1 else (35, 35, 220)
        enemy_color = (35, 35, 220) if camp == 1 else (220, 105, 35)
        for rtype, x, y, hp_frac in _entities(state, ix, "ally"):
            x, y = _actual_xy(x, y, camp)
            if abs(x - ex) < 1e-3 and abs(y - ey) < 1e-3:
                continue
            p = point(x, y)
            cv2.circle(image, p, 9, own_color, -1, cv2.LINE_AA)
            cv2.putText(image, f"{own_label[0]} {SHORT_NAME[rtype]} {hp_frac:.0%}",
                        (p[0] + 10, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.39,
                        own_color, 1, cv2.LINE_AA)
        shown_enemies = []
        for rtype, x, y, hp_frac in enemies:
            x, y = _actual_xy(x, y, camp)
            p = point(x, y)
            shown_enemies.append((rtype, x, y, hp_frac))
            cv2.drawMarker(image, p, enemy_color, cv2.MARKER_TILTED_CROSS,
                           16, 2, cv2.LINE_AA)
            cv2.putText(image, f"{enemy_label[0]} {SHORT_NAME[rtype]} {hp_frac:.0%}",
                        (p[0] + 10, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.39,
                        enemy_color, 1, cv2.LINE_AA)
        cv2.drawMarker(image, ego_point, own_color, cv2.MARKER_STAR,
                       19, 2, cv2.LINE_AA)
        cv2.putText(image, f"{own_label} IQL ego {float(state[ix['ego.hp_frac']]):.0%}",
                    (ego_point[0] + 12, ego_point[1] + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.43, own_color, 1, cv2.LINE_AA)

        dx, dy = action[:2] * scale[:2]
        if camp == 1:
            dx, dy = -dx, -dy
        goal_point = point(np.clip(ex + dx, 0, S.FIELD_X),
                           np.clip(ey + dy, 0, S.FIELD_Y))
        cv2.arrowedLine(image, ego_point, goal_point, (0, 135, 230), 3,
                        cv2.LINE_AA, tipLength=0.12)
        target_text, target = _target_name(action)
        if target != NO_TARGET:
            wanted = S.MOBILE_TYPES[target]
            for rtype, tx, ty, _ in shown_enemies:
                if rtype == wanted:
                    dashed(image, ego_point, point(tx, ty), (0, 135, 230))
                    break

        fire = "ON" if action[2] > 0.5 else "off"
        hp = lambda name: float(state[ix[name]])
        own_base, own_outpost = hp("build.own.base_hp"), hp("build.own.outpost_hp")
        enemy_base, enemy_outpost = hp("build.enemy.base_hp"), hp("build.enemy.outpost_hp")
        if camp == 1:
            red_base, red_outpost = enemy_base, enemy_outpost
            blue_base, blue_outpost = own_base, own_outpost
        else:
            red_base, red_outpost = own_base, own_outpost
            blue_base, blue_outpost = enemy_base, enemy_outpost
        for name, value, color in (
                ("RED base", red_base, (35, 35, 220)),
                ("BLUE base", blue_base, (220, 105, 35)),
                ("RED outpost", red_outpost, (35, 35, 220)),
                ("BLUE outpost", blue_outpost, (220, 105, 35))):
            key = name.lower().replace(" ", "_")
            p = point(*RMUC2026_LANDMARKS[key])
            cv2.rectangle(image, (p[0] - 8, p[1] - 8), (p[0] + 8, p[1] + 8), color, -1)
            cv2.putText(image, f"{name} {value:.0%}", (p[0] + 10, p[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.41, color, 1, cv2.LINE_AA)

        title = (f"OfflineRL IQL | held-out match {game_id} | t={frame + 1:03d}s | "
                 f"controlled camp: {own_label}")
        action_text = (f"IQL: goal ({dx:+.1f}, {dy:+.1f}) m | fire {fire} | "
                       f"target {target_text}")
        hp_text = (f"ego heat {float(state[ix['ego.heat_frac']]):.0%} | "
                   f"own coin {float(state[ix['econ.coin_total']]) * 4000:.0f} | "
                   f"enemy coin {float(state[ix['econ.enemy_coin_total']]) * 4000:.0f}")
        cv2.putText(image, title, (45, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                    (35, 45, 55), 2, cv2.LINE_AA)
        cv2.putText(image, "orange arrow: IQL 5 s goal | orange dashed: selected target | dots/crosses: recorded vehicles", (45, 74),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (70, 80, 90), 1, cv2.LINE_AA)
        cv2.putText(image, action_text, (45, 820), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (0, 105, 185), 2, cv2.LINE_AA)
        cv2.putText(image, hp_text, (45, 846), cv2.FONT_HERSHEY_SIMPLEX, 0.56,
                    (45, 55, 65), 1, cv2.LINE_AA)
        writer.write(image)
    writer.release()
    return len(frame_ids)


def run(data_dir: str, run_dir: str, out: str, episode: int | None,
        fps: int = 12, stride: int = 1, device: str = "cpu",
        renderer: str = "matplotlib", map_image: str | None = None):
    z = np.load(os.path.join(data_dir, "val.npz"))
    names = obs_feature_names()
    ix = _index(names)
    ep = _choose_episode(z, ix) if episode is None else episode
    if ep < 0 or ep >= len(z["ep_start"]):
        raise ValueError(f"episode must be in [0, {len(z['ep_start']) - 1}]")
    start, length = int(z["ep_start"][ep]), int(z["ep_len"][ep])
    raw = z["obs"][start:start + length]
    game_id = int(z["ep_game_id"][ep])
    camp = int(z["ep_camp"][ep])
    with open(os.path.join(data_dir, "meta.json"), encoding="utf-8") as fh:
        scale = np.asarray(json.load(fh)["act_scale"], dtype=np.float32)

    model, norm, info = load_policy(run_dir, device=device)
    if info["action_mode"] != "tactical":
        raise ValueError("offline replay requires a tactical-action policy")
    pred = _policy_actions(model, norm, raw, device)

    base, ext = os.path.splitext(out)
    if ext.lower() != ".mp4":
        raise ValueError("--out must end in .mp4")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    storyboard = base + "_storyboard.png"
    _save_storyboard(storyboard, raw, pred, scale, ix, game_id)

    if renderer == "opencv":
        displayed_frames = _render_continuous_opencv(
            out, raw, pred, scale, ix, game_id, camp, fps, stride, map_image)
    else:
        fig = plt.figure(figsize=(12.8, 7.2), constrained_layout=True)
        grid = fig.add_gridspec(2, 1, height_ratios=(4, 1.2))
        field_ax = fig.add_subplot(grid[0])
        hp_ax = fig.add_subplot(grid[1])
        frames = np.arange(0, len(raw), max(1, stride), dtype=int)

        def update(frame_no):
            frame = int(frames[frame_no])
            field_ax.clear()
            _draw_state(field_ax, raw[frame], pred[frame], scale, ix, frame + 1, game_id)
            _draw_hp(hp_ax, raw, ix, frame)

        anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps,
                             blit=False, repeat=True)
        anim.save(out, writer=FFMpegWriter(fps=fps, bitrate=2600))
        plt.close(fig)
        displayed_frames = len(frames)

    manifest = dict(
        source="held-out official match replay; IQL action overlay",
        episode_index=ep,
        game_id=game_id,
        transitions=int(length),
        displayed_frames=int(displayed_frames),
        fps=fps,
        stride_seconds=stride,
        renderer=renderer,
        map_image=os.path.basename(map_image) if map_image else None,
        video=os.path.basename(out),
        storyboard=os.path.basename(storyboard),
    )
    with open(base + ".json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser(description="Render a complete offlineRL IQL replay")
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="rm_runs/sparring_visuals/offline_iql_replay.mp4")
    ap.add_argument("--episode", type=int, default=None,
                    help="held-out episode index; default selects an active one")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--stride", type=int, default=1,
                    help="display one frame every N recorded seconds")
    ap.add_argument("--renderer", choices=["matplotlib", "opencv"], default="matplotlib",
                    help="opencv is faster for every-second continuous MP4 output")
    ap.add_argument("--map-image", default=None,
                    help="optional RMUC occupancy/semantic map used as the replay background")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    run(args.data, args.run, args.out, args.episode, args.fps, args.stride,
        args.device, args.renderer, args.map_image)


if __name__ == "__main__":
    main()
