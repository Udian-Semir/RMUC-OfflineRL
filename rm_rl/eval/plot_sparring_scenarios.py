"""Visualise held-out 2D situations for the offline sparring policies.

This is deliberately a *one-step*, held-out inspection tool.  It does not
pretend the logged positions form a closed-loop physics rollout: the offline
policies model what a teammate/opponent would request at the next tactical
second, while the future positions in the log still come from the real match.
That distinction matters before this policy is connected to the online sentry
environment.

Example:
    python -m rm_rl.eval.plot_sparring_scenarios \
        --data data/infantry_tactical \
        --iql-run rm_runs/infantry_iql_tactical \
        --bc-run rm_runs/infantry_bc_tactical \
        --out rm_runs/sparring_visuals/heldout_scenarios.png
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

# The default home is read-only in some sandboxed runs.  Set this before pyplot
# initialises its font/cache directories.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-rmrl")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

from ..algos.action_spec import NO_TARGET
from ..data import schema as S
from ..data.features import obs_feature_names
from ..deploy import load_policy


SHORT_NAME = {
    S.TYPE_HERO: "hero",
    S.TYPE_ENGINEER: "eng",
    S.TYPE_INFANTRY3: "inf3",
    S.TYPE_INFANTRY4: "inf4",
    S.TYPE_AERIAL: "air",
    S.TYPE_SENTRY: "sentry",
}


@dataclass(frozen=True)
class Scene:
    title: str
    index: int
    note: str


def _index(names: list[str]) -> dict[str, int]:
    return {name: i for i, name in enumerate(names)}


def _ego_xy(obs: np.ndarray, ix: dict[str, int]) -> tuple[float, float]:
    return ((float(obs[ix["ego.x"]]) + 1.0) * S.FIELD_X / 2.0,
            (float(obs[ix["ego.y"]]) + 1.0) * S.FIELD_Y / 2.0)


def _entities(obs: np.ndarray, ix: dict[str, int], side: str):
    """Return known mobile positions reconstructed from relative feature slots."""
    ex, ey = _ego_xy(obs, ix)
    prefix = "ally" if side == "ally" else "enemy"
    out = []
    for rtype in S.MOBILE_TYPES:
        base = f"{prefix}[{rtype}]"
        alive = float(obs[ix[f"{base}.alive"]]) > 0.5
        known = alive if side == "ally" else float(obs[ix[f"{base}.pos_known"]]) > 0.5
        if not known:
            continue
        x = ex + float(obs[ix[f"{base}.rx"]]) * S.FIELD_X
        y = ey + float(obs[ix[f"{base}.ry"]]) * S.FIELD_Y
        if -0.25 <= x <= S.FIELD_X + 0.25 and -0.25 <= y <= S.FIELD_Y + 0.25:
            out.append((rtype, x, y, float(obs[ix[f"{base}.hp_frac"]])))
    return out


def _model_action(model, normalizer, obs: np.ndarray, device: str) -> np.ndarray:
    with torch.no_grad():
        x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        return model.act(normalizer.normalize(x), deterministic=True)[0].cpu().numpy()


def _choose_scenes(obs: np.ndarray, act: np.ndarray, valid: np.ndarray,
                   ix: dict[str, int], scale: np.ndarray) -> list[Scene]:
    """Choose three interpretable cases from the held-out trajectories."""
    valid = valid > 0.5
    elapsed = obs[:, ix["time.elapsed"]]
    nav = np.linalg.norm(act[:, :2] * scale[:2], axis=1)
    logged_target = act[:, 3:].argmax(axis=1)
    # Prefer frames where the held-out demonstrator is actually in an
    # interaction: this makes the target and fire heads inspectable in the
    # picture.  The semantic condition below is still the primary filter.
    combat = (act[:, 2] > 0.5) & (logged_target != NO_TARGET)

    # The exact rule requested for the sparring context: after the first minute
    # the enemy outpost is still alive.  Prefer a genuine movement decision.
    after_minute = (valid & (elapsed >= 60.0 / 420.0)
                    & (obs[:, ix["build.enemy.outpost_alive"]] > 0.5))
    a = np.flatnonzero(after_minute & combat)
    if not len(a):
        a = np.flatnonzero(after_minute)
    i_outpost = int(a[np.argmax(nav[a])]) if len(a) else int(np.flatnonzero(valid)[0])

    # A 25% base fraction is an unambiguous pressure state.  Pick an active
    # engagement there when the held-out record provides one.
    own_base = obs[:, ix["build.own.base_hp"]]
    under_pressure = valid & (own_base < 0.25)
    b = np.flatnonzero(under_pressure & combat)
    if not len(b):
        b = np.flatnonzero(under_pressure)
    i_base = int(b[np.argmin(own_base[b])]) if len(b) else int(np.flatnonzero(valid)[np.argmin(own_base[valid])])

    # "Allies push" is intentionally defined only from coordinates, matching
    # the radar-side state specification rather than adding an opaque label.
    ex = (obs[:, ix["ego.x"]] + 1.0) * S.FIELD_X / 2.0
    max_ally_x = np.full(len(obs), -np.inf, np.float32)
    for rtype in S.MOBILE_TYPES:
        base = f"ally[{rtype}]"
        alive = obs[:, ix[f"{base}.alive"]] > 0.5
        x = ex + obs[:, ix[f"{base}.rx"]] * S.FIELD_X
        # Do not score the ego slot itself as a teammate push.
        non_ego = np.abs(obs[:, ix[f"{base}.rx"]]) + np.abs(obs[:, ix[f"{base}.ry"]]) > 1e-4
        max_ally_x = np.maximum(max_ally_x, np.where(alive & non_ego, x, -np.inf))
    pushed = valid & (max_ally_x > S.FIELD_X * 0.55)
    c = np.flatnonzero(pushed & combat)
    if not len(c):
        c = np.flatnonzero(pushed)
    i_push = int(c[np.argmax(max_ally_x[c])]) if len(c) else i_outpost

    return [
        Scene("After 60 s: enemy outpost still alive", i_outpost,
              "pressure opportunity; outpost has not fallen"),
        Scene("Own base under pressure", i_base,
              "low base HP with a held-out active engagement"),
        Scene("Allied forward push", i_push,
              "teammate advancement inferred only from coordinates"),
    ]


def _action_text(action: np.ndarray, scale: np.ndarray) -> tuple[str, int]:
    dx, dy = action[:2] * scale[:2]
    fire = "on" if action[2] > 0.5 else "off"
    tgt = int(np.argmax(action[3:]))
    target = "none" if tgt == NO_TARGET else SHORT_NAME[S.MOBILE_TYPES[tgt]]
    return f"goal ({dx:+.1f}, {dy:+.1f})  fire {fire}  target {target}", tgt


def _draw_arrow(ax, x: float, y: float, action: np.ndarray, scale: np.ndarray,
                color: str, label: str, offset: float = 0.0):
    dx, dy = action[:2] * scale[:2]
    # Offset roots very slightly so three identical suggestions remain visible.
    ax.arrow(x + offset, y - offset, dx, dy, width=0.045, head_width=0.42,
             head_length=0.50, length_includes_head=True, color=color,
             alpha=0.9, zorder=5, label=label)


def _draw_target(ax, x: float, y: float, target: int, enemy, color: str):
    if target == NO_TARGET:
        return
    wanted = S.MOBILE_TYPES[target]
    for rtype, tx, ty, _ in enemy:
        if rtype == wanted:
            ax.plot([x, tx], [y, ty], linestyle=(0, (3, 3)), color=color,
                    linewidth=1.1, alpha=0.8, zorder=3)
            return


def _draw_scene(ax, scene: Scene, obs: np.ndarray, recorded: np.ndarray,
                iql: np.ndarray, bc: np.ndarray, scale: np.ndarray,
                ix: dict[str, int]):
    # This is a metric field reference, not a claim that we recovered the
    # competition mesh.  Obstacles will come from the semantic map during the
    # later online-environment integration.
    ax.set_facecolor("#f8fafc")
    ax.add_patch(plt.Rectangle((0, 0), S.FIELD_X, S.FIELD_Y, fill=False,
                               linewidth=2.0, edgecolor="#243447"))
    ax.axvline(S.FIELD_X / 2, color="#93a1a1", linewidth=1.0, linestyle="--")
    ax.axvspan(0, S.FIELD_X / 2, color="#fee2e2", alpha=0.30, zorder=0)
    ax.axvspan(S.FIELD_X / 2, S.FIELD_X, color="#dbeafe", alpha=0.28, zorder=0)

    # Fixed strategic landmarks make HP semantics visible without inventing a
    # detailed obstacle map that is not present in the offline referee logs.
    for x, name, own in ((1.2, "own base", True), (7.0, "own outpost", True),
                         (21.0, "enemy outpost", False), (26.8, "enemy base", False)):
        color = "#991b1b" if own else "#1d4ed8"
        ax.scatter([x], [S.FIELD_Y / 2], marker="s", s=75, color=color, zorder=2)
        ax.text(x, S.FIELD_Y / 2 + 0.6, name, ha="center", va="bottom", fontsize=7, color=color)

    ex, ey = _ego_xy(obs, ix)
    allies = _entities(obs, ix, "ally")
    enemies = _entities(obs, ix, "enemy")
    for rtype, x, y, hp in allies:
        if abs(x - ex) < 1e-3 and abs(y - ey) < 1e-3:
            continue
        ax.scatter([x], [y], s=42, color="#dc2626", zorder=4)
        ax.text(x + 0.22, y + 0.22, f"{SHORT_NAME[rtype]} {hp:.0%}", fontsize=6.5, color="#7f1d1d")
    for rtype, x, y, hp in enemies:
        ax.scatter([x], [y], s=48, marker="X", color="#2563eb", zorder=4)
        ax.text(x + 0.22, y + 0.22, f"{SHORT_NAME[rtype]} {hp:.0%}", fontsize=6.5, color="#1e3a8a")
    ax.scatter([ex], [ey], s=180, marker="*", color="#b91c1c", edgecolor="white",
               linewidth=0.7, zorder=6)
    ax.text(ex + 0.25, ey - 0.55, "ego", fontsize=7, color="#7f1d1d", weight="bold")

    # Log, BC, and IQL commands share exactly the same held-out state.
    _draw_arrow(ax, ex, ey, recorded, scale, "#64748b", "logged", -0.10)
    _draw_arrow(ax, ex, ey, bc, scale, "#7c3aed", "BC", 0.00)
    _draw_arrow(ax, ex, ey, iql, scale, "#d97706", "IQL", 0.10)
    _, lt = _action_text(recorded, scale)
    _, bt = _action_text(bc, scale)
    _, it = _action_text(iql, scale)
    _draw_target(ax, ex, ey, lt, enemies, "#64748b")
    _draw_target(ax, ex, ey, bt, enemies, "#7c3aed")
    _draw_target(ax, ex, ey, it, enemies, "#d97706")

    time_s = float(obs[ix["time.elapsed"]]) * 420.0
    hp = lambda name: float(obs[ix[name]])
    status = (f"t={time_s:.0f}s | own base {hp('build.own.base_hp'):.0%}, "
              f"outpost {'alive' if hp('build.own.outpost_alive') > .5 else 'down'} | "
              f"enemy base {hp('build.enemy.base_hp'):.0%}, "
              f"outpost {'alive' if hp('build.enemy.outpost_alive') > .5 else 'down'}")
    ax.set_title(f"{scene.title}\n{scene.note}", fontsize=10, loc="left", pad=8)
    ax.text(0.0, -0.23, status + "\n"
            + "log: " + _action_text(recorded, scale)[0] + "\n"
            + "BC:  " + _action_text(bc, scale)[0] + "\n"
            + "IQL: " + _action_text(iql, scale)[0],
            transform=ax.transAxes, fontsize=7.3, va="top", family="monospace")
    ax.set_xlim(-0.5, S.FIELD_X + 0.5)
    ax.set_ylim(-0.5, S.FIELD_Y + 0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([0, 7, 14, 21, 28])
    ax.set_yticks([0, 7.5, 15])
    ax.tick_params(labelsize=7)
    ax.set_xlabel("field x (m), own half -> enemy half", fontsize=7)


def run(data_dir: str, iql_run: str, bc_run: str, out: str, device: str = "cpu"):
    data = np.load(os.path.join(data_dir, "val.npz"))
    obs, act, valid = data["obs"], data["act"], data["valid"]
    with open(os.path.join(data_dir, "meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    scale = np.asarray(meta["act_scale"], dtype=np.float32)
    ix = _index(obs_feature_names())
    scenes = _choose_scenes(obs, act, valid, ix, scale)

    iql_model, iql_norm, iql_info = load_policy(iql_run, device=device)
    bc_model, bc_norm, bc_info = load_policy(bc_run, device=device)
    if iql_info["action_mode"] != "tactical" or bc_info["action_mode"] != "tactical":
        raise ValueError("scenario visualisation requires tactical-action checkpoints")
    if not np.allclose(iql_info["act_scale"], scale) or not np.allclose(bc_info["act_scale"], scale):
        raise ValueError("checkpoint action scale does not match the held-out dataset")

    fig, axes = plt.subplots(1, len(scenes), figsize=(18, 5.8), constrained_layout=True)
    for ax, scene in zip(np.atleast_1d(axes), scenes):
        raw = obs[scene.index]
        iql = _model_action(iql_model, iql_norm, raw, device)
        bc = _model_action(bc_model, bc_norm, raw, device)
        _draw_scene(ax, scene, raw, act[scene.index], iql, bc, scale, ix)
    handles = [
        Line2D([0], [0], color="#64748b", lw=2, label="logged action"),
        Line2D([0], [0], color="#7c3aed", lw=2, label="BC policy"),
        Line2D([0], [0], color="#d97706", lw=2, label="IQL policy"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#b91c1c", markersize=10, label="controlled unit"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor="#2563eb", markersize=8, label="known enemy"),
    ]
    fig.legend(handles=handles, ncols=5, loc="upper center", bbox_to_anchor=(0.5, 1.04), fontsize=8)
    fig.suptitle("Held-out 2D sparring-policy inspection (one-step, canonical radar frame)",
                 fontsize=12, y=1.08)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    selection = [{"title": s.title, "val_transition_index": s.index, "note": s.note}
                 for s in scenes]
    with open(os.path.splitext(out)[0] + ".json", "w", encoding="utf-8") as fh:
        json.dump(selection, fh, ensure_ascii=False, indent=2)
    print(f"saved {out}")
    for s in selection:
        print(f"  {s['title']}: val transition {s['val_transition_index']}")


def main():
    ap = argparse.ArgumentParser(description="Plot held-out 2D offline sparring scenes")
    ap.add_argument("--data", required=True)
    ap.add_argument("--iql-run", required=True)
    ap.add_argument("--bc-run", required=True)
    ap.add_argument("--out", default="rm_runs/sparring_visuals/heldout_scenarios.png")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    run(args.data, args.iql_run, args.bc_run, args.out, args.device)


if __name__ == "__main__":
    main()
