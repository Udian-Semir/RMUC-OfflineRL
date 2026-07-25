"""Open one simple map figure for the trained offlineRL IQL policy.

The terminal prints the detailed statistics.  The window intentionally only
contains the RMUC map, the recorded ego route, IQL's predicted goal points, and
the red/blue units with HP at one chosen second.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import torch

from ..algos.action_spec import NO_TARGET
from ..data import schema as S
from ..data.features import obs_feature_names
from ..deploy import load_policy
from .plot_sparring_scenarios import SHORT_NAME, _ego_xy, _entities, _index
from .render_offline_replay import RMUC2026_LANDMARKS, _actual_xy, _choose_episode


def _actions(model, norm, obs: np.ndarray, device: str):
    with torch.no_grad():
        x = torch.as_tensor(obs, dtype=torch.float32, device=device)
        return model.act(norm.normalize(x), deterministic=True).cpu().numpy()


def _match_info(db: str, game_id: int):
    with sqlite3.connect(db) as con:
        row = con.execute(
            'SELECT "赛区", "红方学校", "蓝方学校", "胜方", "时长秒", "开始时间" '
            'FROM matches WHERE game_id=?', (game_id,)).fetchone()
    return row


def _building_rows(state: np.ndarray, ix: dict[str, int], camp: int):
    own_base, own_out = state[ix["build.own.base_hp"]], state[ix["build.own.outpost_hp"]]
    enemy_base, enemy_out = state[ix["build.enemy.base_hp"]], state[ix["build.enemy.outpost_hp"]]
    return (("RED base", own_base if camp == 0 else enemy_base, "red_base", "#dc2626"),
            ("BLUE base", own_base if camp == 1 else enemy_base, "blue_base", "#2563eb"),
            ("RED outpost", own_out if camp == 0 else enemy_out, "red_outpost", "#dc2626"),
            ("BLUE outpost", own_out if camp == 1 else enemy_out, "blue_outpost", "#2563eb"))


def _animate(ax, raw, pred, dxy, ego_xy, ix, camp, game_id, interval_ms):
    """Update one official recorded second per frame with the IQL decision."""
    camp_name = "BLUE" if camp == 1 else "RED"
    own_name, enemy_name = camp_name, "RED" if camp == 1 else "BLUE"
    own_color, enemy_color = ("#2563eb", "#dc2626") if camp == 1 else ("#dc2626", "#2563eb")
    route, = ax.plot([], [], color=own_color, linewidth=1.5, alpha=.72,
                     label="recorded ego route", zorder=2)
    status = ax.text(.01, 1.015, "", transform=ax.transAxes, fontsize=9,
                     color="#374151", va="bottom")
    terminal_frame = [-1]
    dynamic = []

    def add(artist):
        dynamic.append(artist)
        return artist

    def update(frame):
        for artist in dynamic:
            artist.remove()
        dynamic.clear()
        route.set_data(ego_xy[:frame + 1, 0], ego_xy[:frame + 1, 1])
        state, action = raw[frame], pred[frame]
        ex, ey = ego_xy[frame]
        enemies = []
        for rtype, x, y, hp in _entities(state, ix, "ally"):
            x, y = _actual_xy(x, y, camp)
            if abs(x - ex) < 1e-3 and abs(y - ey) < 1e-3:
                continue
            add(ax.scatter([x], [y], s=58, color=own_color, edgecolor="white",
                           linewidth=.5, zorder=5))
            add(ax.text(x + .15, y + .15, f"{own_name[0]} {SHORT_NAME[rtype]} {hp:.0%}",
                        fontsize=7, color=own_color, zorder=6))
        for rtype, x, y, hp in _entities(state, ix, "enemy"):
            x, y = _actual_xy(x, y, camp)
            enemies.append((rtype, x, y))
            add(ax.scatter([x], [y], s=62, marker="X", color=enemy_color, zorder=5))
            add(ax.text(x + .15, y + .15, f"{enemy_name[0]} {SHORT_NAME[rtype]} {hp:.0%}",
                        fontsize=7, color=enemy_color, zorder=6))
        add(ax.scatter([ex], [ey], s=195, marker="*", color=own_color,
                       edgecolor="white", linewidth=.7, zorder=7))
        add(ax.text(ex + .18, ey - .35,
                    f"{own_name} IQL ego {state[ix['ego.hp_frac']]:.0%}",
                    fontsize=7, color=own_color, zorder=8))

        dx, dy = dxy[frame]
        goal = np.clip([ex + dx, ey + dy], [0.0, 0.0], [S.FIELD_X, S.FIELD_Y])
        add(ax.annotate("", xy=goal, xytext=(ex, ey),
                        arrowprops=dict(arrowstyle="->", color="#d97706", lw=2.3),
                        zorder=8))
        add(ax.scatter([goal[0]], [goal[1]], marker="x", s=75, color="#d97706", zorder=8))
        target = int(action[3:].argmax())
        target_name = "none" if target == NO_TARGET else SHORT_NAME[S.MOBILE_TYPES[target]]
        if target != NO_TARGET:
            wanted = S.MOBILE_TYPES[target]
            for rtype, tx, ty in enemies:
                if rtype == wanted:
                    add(ax.plot([ex, tx], [ey, ty], color="#d97706", linestyle=(0, (3, 3)),
                                linewidth=1.4, zorder=4)[0])
                    break
        for label, hp, key, color in _building_rows(state, ix, camp):
            x, y = RMUC2026_LANDMARKS[key]
            add(ax.scatter([x], [y], marker="s", s=55, color=color, zorder=5))
            add(ax.text(x + .15, y + .15, f"{label} {hp:.0%}", fontsize=7,
                        color=color, zorder=6))

        fire = bool(action[2] > .5)
        text = (f"match {game_id} | t={frame + 1:03d}s | camp {camp_name} | "
                f"IQL goal ({dx:+.2f}, {dy:+.2f}) | fire {fire} | target {target_name}")
        status.set_text(text)
        if terminal_frame[0] != frame:
            print("[decision] " + text + f" | ego_hp={state[ix['ego.hp_frac']]:.1%}", flush=True)
            terminal_frame[0] = frame
        return [route, status, *dynamic]

    animation = FuncAnimation(ax.figure, update, frames=len(raw), interval=interval_ms,
                              blit=False, repeat=False, cache_frame_data=False)
    ax.figure._offline_animation = animation  # retain until the GUI window closes


def run(data: str, run_dir: str, map_image: str, db: str, episode: int | None,
        second: int, sample: int, animate: bool, interval_ms: int,
        device: str, out: str | None):
    z = np.load(os.path.join(data, "val.npz"))
    ix = _index(obs_feature_names())
    ep = _choose_episode(z, ix) if episode is None else episode
    if ep < 0 or ep >= len(z["ep_start"]):
        raise ValueError(f"episode must be in [0, {len(z['ep_start']) - 1}]")
    start, length = int(z["ep_start"][ep]), int(z["ep_len"][ep])
    second = int(np.clip(second, 1, length))
    raw, logged = z["obs"][start:start + length], z["act"][start:start + length]
    camp, game_id = int(z["ep_camp"][ep]), int(z["ep_game_id"][ep])
    with open(os.path.join(data, "meta.json"), encoding="utf-8") as fh:
        scale = np.asarray(json.load(fh)["act_scale"], np.float32)
    model, norm, info = load_policy(run_dir, device=device)
    if info["action_mode"] != "tactical":
        raise ValueError("this viewer requires a tactical IQL/BC checkpoint")
    pred = _actions(model, norm, raw, device)

    # Policy uses the canonical own-left frame.  Undo it for blue episodes so
    # the plot uses the real red/blue orientation of the supplied RMUC map.
    ego_xy = np.array([_actual_xy(*_ego_xy(s, ix), camp) for s in raw])
    dxy = pred[:, :2] * scale[:2]
    if camp == 1:
        dxy *= -1.0
    goals = np.clip(ego_xy + dxy, [0.0, 0.0], [S.FIELD_X, S.FIELD_Y])
    fire_rate = float((pred[:, 2] > 0.5).mean())
    named_rate = float((pred[:, 3:].argmax(1) != NO_TARGET).mean())
    log_fire_rate = float((logged[:, 2] > 0.5).mean())
    m = _match_info(db, game_id)
    camp_name = "BLUE" if camp == 1 else "RED"
    print(f"match={game_id} region={m[0]} red={m[1]} blue={m[2]} winner={m[3]} "
          f"duration={m[4]} start={m[5]}")
    print(f"heldout episode={ep} controlled_camp={camp_name} seconds={length}")
    print(f"IQL predicted_fire_rate={fire_rate:.1%} "
          f"predicted_named_target_rate={named_rate:.1%} "
          f"logged_fire_rate={log_fire_rate:.1%}")

    image = plt.imread(map_image)
    fig, ax = plt.subplots(figsize=(15, 8.3))
    ax.imshow(image, cmap="gray", extent=[0, S.FIELD_X, 0, S.FIELD_Y],
              origin="upper", zorder=0)
    if animate:
        _animate(ax, raw, pred, dxy, ego_xy, ix, camp, game_id, interval_ms)
    else:
        ax.plot(ego_xy[:, 0], ego_xy[:, 1], color="#2563eb" if camp == 1 else "#dc2626",
                linewidth=1.5, alpha=0.65, label="recorded ego route", zorder=2)
        ax.scatter(goals[::sample, 0], goals[::sample, 1], marker="x", s=24, color="#d97706",
                   alpha=0.55, label="IQL predicted 5 s goals", zorder=3)

    if not animate:
        state, action = raw[second - 1], pred[second - 1]
        own_name, enemy_name = camp_name, "RED" if camp == 1 else "BLUE"
        own_color, enemy_color = ("#2563eb", "#dc2626") if camp == 1 else ("#dc2626", "#2563eb")
        ex, ey = ego_xy[second - 1]
        for rtype, x, y, hp in _entities(state, ix, "ally"):
            x, y = _actual_xy(x, y, camp)
            if abs(x - ex) < 1e-3 and abs(y - ey) < 1e-3:
                continue
            ax.scatter([x], [y], s=56, color=own_color, edgecolor="white", linewidth=0.5, zorder=5)
            ax.text(x + .15, y + .15, f"{own_name[0]} {SHORT_NAME[rtype]} {hp:.0%}",
                    fontsize=7, color=own_color, zorder=6)
        enemies = []
        for rtype, x, y, hp in _entities(state, ix, "enemy"):
            x, y = _actual_xy(x, y, camp)
            enemies.append((rtype, x, y))
            ax.scatter([x], [y], s=58, marker="X", color=enemy_color, zorder=5)
            ax.text(x + .15, y + .15, f"{enemy_name[0]} {SHORT_NAME[rtype]} {hp:.0%}",
                    fontsize=7, color=enemy_color, zorder=6)
        ax.scatter([ex], [ey], s=190, marker="*", color=own_color, edgecolor="white",
                   linewidth=.7, zorder=7)
        ax.text(ex + .18, ey - .35, f"{own_name} IQL ego {state[ix['ego.hp_frac']]:.0%}",
                fontsize=7, color=own_color, zorder=8)

        target = int(action[3:].argmax())
        if target != NO_TARGET:
            wanted = S.MOBILE_TYPES[target]
            for rtype, tx, ty in enemies:
                if rtype == wanted:
                    ax.plot([ex, tx], [ey, ty], color="#d97706", linestyle=(0, (3, 3)),
                            linewidth=1.4, zorder=4)
                    break
        for label, hp, key, color in _building_rows(state, ix, camp):
            x, y = RMUC2026_LANDMARKS[key]
            ax.scatter([x], [y], marker="s", s=50, color=color, zorder=5)
            ax.text(x + .15, y + .15, f"{label} {hp:.0%}", fontsize=7, color=color, zorder=6)
        target_name = "none" if target == NO_TARGET else SHORT_NAME[S.MOBILE_TYPES[target]]
        dx, dy = dxy[second - 1]
        print(f"t={second}s IQL goal=({dx:+.2f},{dy:+.2f}) fire={bool(action[2] > .5)} "
              f"target={target_name} ego_hp={state[ix['ego.hp_frac']]:.1%}")
        ax.set_title(f"OfflineRL IQL: held-out match {game_id}, t={second}s, controlled camp {camp_name}")
    ax.set_xlim(0, S.FIELD_X)
    ax.set_ylim(0, S.FIELD_Y)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if not animate:
        ax.legend(loc="upper center", ncol=2, fontsize=8)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=180)
        print(f"saved {out}")
    plt.show()


def main():
    ap = argparse.ArgumentParser(description="Show offlineRL IQL points on the RMUC map")
    ap.add_argument("--data", default="data/infantry_tactical")
    ap.add_argument("--run", default="rm_runs/infantry_iql_tactical")
    ap.add_argument(
        "--map",
        default="/home/julia/workspace/Radar-Station/src/battlefield_visualizer/config/2026RMUC.png",
        help="landscape RMUC map; use the unrotated image for the 28 x 15 m log frame",
    )
    ap.add_argument("--db", default="dataset/rmuc_2026_region_dataset.sqlite")
    ap.add_argument("--episode", type=int, default=None)
    ap.add_argument("--time", type=int, default=152, help="shown recorded second")
    ap.add_argument("--sample", type=int, default=4,
                    help="draw one IQL goal point every N seconds")
    ap.add_argument("--animate", action="store_true",
                    help="update the map every recorded second and print each decision")
    ap.add_argument("--interval-ms", type=int, default=250,
                    help="wall-clock delay per recorded second in --animate mode")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="optional PNG output in addition to the window")
    args = ap.parse_args()
    run(args.data, args.run, args.map, args.db, args.episode, args.time,
        max(1, args.sample), args.animate, max(1, args.interval_ms), args.device, args.out)


if __name__ == "__main__":
    main()
