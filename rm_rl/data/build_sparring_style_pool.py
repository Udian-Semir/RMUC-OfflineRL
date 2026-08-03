"""Create an auditable multi-style sparring pool from official RMUC logs.

Styles are assigned per (game, camp), never per individual second.  This
prevents a policy from learning one half of a team's match while its held-out
evaluation contains the other half.  The source role datasets already hold
whole-episode game/camp identifiers, so this tool only partitions episodes and
does not invent actions.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from . import schema as S

STYLES = ("aggressive", "pressure", "measured")
ROLES = ("hero", "engineer", "infantry", "aerial", "sentry")


def _z(value: np.ndarray) -> np.ndarray:
    return (value - value.mean()) / max(float(value.std()), 1e-6)


def _match_features(con: sqlite3.Connection, game_id: int) -> list[dict]:
    q = '''SELECT "时刻秒" AS t, "机器人类型" AS rtype, "阵营" AS camp,
                  x, y, "当前血量" AS hp, "最大血量" AS maxhp,
                  "累计17mm发弹" AS ammo17, "累计42mm发弹" AS ammo42
           FROM timeseries WHERE game_id = ?'''
    df = pd.read_sql(q, con, params=[int(game_id)])
    out = []
    for camp in S.CAMPS:
        own = df[(df.camp == camp) & df.rtype.isin(S.MOBILE_TYPES)].copy()
        if own.empty:
            continue
        own["canonical_x"] = own.x if camp == S.CAMP_RED else S.FIELD_X - own.x
        alive = own[own.hp > 0]
        advance = float(np.clip(alive.canonical_x.mean() / S.FIELD_X if not alive.empty else 0.0, 0.0, 1.0))
        survival = float(np.clip((alive.hp / alive.maxhp.clip(lower=1)).mean() if not alive.empty else 0.0, 0.0, 1.0))
        shots = 0.0
        for _, unit in own.groupby(["rtype"]):
            unit = unit.sort_values("t")
            shots += float(np.clip(unit.ammo17.diff().fillna(0), 0, None).sum())
            shots += float(np.clip(unit.ammo42.diff().fillna(0), 0, None).sum())
        duration = max(float(own.t.max() - own.t.min() + 1), 1.0)
        enemy = S.enemy_camp(camp)
        structures = df[(df.camp == enemy) & df.rtype.isin(S.BUILDING_TYPES)]
        objective_damage = 0.0
        for _, building in structures.groupby("rtype"):
            hp = building.sort_values("t").hp.to_numpy(float)
            if hp.size:
                objective_damage += max(0.0, float(hp[0] - hp.min()))
        out.append(dict(game_id=int(game_id), camp=camp, advance=advance,
                        fire_per_s=shots / duration, objective_damage=objective_damage,
                        survival=survival))
    return out


def build_assignments(db: Path) -> list[dict]:
    con = sqlite3.connect(db)
    games = pd.read_sql("SELECT game_id FROM matches", con).game_id.astype(int).tolist()
    rows = []
    for game_id in games:
        rows.extend(_match_features(con, game_id))
    con.close()
    if len(rows) < 9:
        raise RuntimeError("not enough game/camp rows to define three styles")
    frame = pd.DataFrame(rows)
    # pressure means objective-oriented forward play; aggressive means high
    # contact/forward play after pressure cases are removed; measured is the
    # remainder.  Equal-sized bins avoid a tiny style being mistaken for a
    # generally intelligent policy.
    frame["pressure_score"] = _z(frame.objective_damage.to_numpy()) + 0.50 * _z(frame.advance.to_numpy())
    frame["aggression_score"] = _z(frame.fire_per_s.to_numpy()) + _z(frame.advance.to_numpy()) + 0.25 * _z(frame.survival.to_numpy())
    frame["style"] = "measured"
    n = len(frame) // 3
    pressure_idx = frame.sort_values("pressure_score", ascending=False).index[:n]
    frame.loc[pressure_idx, "style"] = "pressure"
    candidates = frame.drop(index=pressure_idx).sort_values("aggression_score", ascending=False).index[:n]
    frame.loc[candidates, "style"] = "aggressive"
    return frame.sort_values(["game_id", "camp"]).to_dict("records")


def _episode_indices(z: np.lib.npyio.NpzFile, assignments: dict[tuple[int, int], str], style: str) -> np.ndarray:
    selected = []
    for idx, (game_id, camp) in enumerate(zip(z["ep_game_id"], z["ep_camp"])):
        if assignments.get((int(game_id), int(camp))) == style:
            selected.append(idx)
    return np.asarray(selected, dtype=np.int64)


def _write_subset(source: Path, target: Path, style: str, assignments: dict[tuple[int, int], str]) -> dict:
    target.mkdir(parents=True, exist_ok=True)
    with (source / "meta.json").open(encoding="utf-8") as handle:
        meta = json.load(handle)
    totals = {}
    for split in ("train", "val"):
        z = np.load(source / f"{split}.npz")
        episodes = _episode_indices(z, assignments, style)
        if not len(episodes):
            raise RuntimeError(f"{source} has no {split} episodes for {style}")
        transition = np.concatenate([
            np.arange(int(z["ep_start"][i]), int(z["ep_start"][i] + z["ep_len"][i])) for i in episodes
        ])
        payload = {key: z[key][transition] for key in ("obs", "next_obs", "act", "rew", "done", "valid")}
        for key in ("ep_return", "ep_win", "ep_weight", "ep_camp", "ep_game_id"):
            payload[key] = z[key][episodes]
        payload["ep_len"] = z["ep_len"][episodes]
        payload["ep_start"] = np.concatenate([[0], np.cumsum(payload["ep_len"])[:-1]]).astype(np.int64)
        np.savez_compressed(target / f"{split}.npz", **payload)
        totals[split] = (int(len(episodes)), int(len(transition)))
        if split == "train":
            obs_mean = payload["obs"].mean(0).astype(np.float32)
            obs_std = np.maximum(payload["obs"].std(0), 1e-3).astype(np.float32)
    source_norm = np.load(source / "norm_stats.npz")
    np.savez(target / "norm_stats.npz", obs_mean=obs_mean, obs_std=obs_std,
             act_scale=source_norm["act_scale"])
    meta.update(
        n_train_episodes=totals["train"][0], n_train_transitions=totals["train"][1],
        n_val_episodes=totals["val"][0], n_val_transitions=totals["val"][1],
        sparring_style=style, style_partition="whole game/camp; source train/val split retained",
        source_dataset=str(source),
    )
    with (target / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
    return {"role": source.name.replace("_tactical_buildings", ""), "style": style,
            "train_episodes": totals["train"][0], "train_transitions": totals["train"][1],
            "val_episodes": totals["val"][0], "val_transitions": totals["val"][1]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--source-root", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("sparring_style_pool"))
    args = ap.parse_args()
    assignments_rows = build_assignments(args.db)
    assignments = {(int(row["game_id"]), 0 if row["camp"] == S.CAMP_RED else 1): row["style"] for row in assignments_rows}
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "style_assignments.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(assignments_rows[0]))
        writer.writeheader(); writer.writerows(assignments_rows)
    counts = {style: sum(row["style"] == style for row in assignments_rows) for style in STYLES}
    (args.out / "style_catalog.json").write_text(json.dumps({
        "styles": list(STYLES), "counts_game_camp": counts,
        "definition": "pressure: high enemy-building damage + advance; aggressive: high fire density + advance among non-pressure; measured: remaining cases",
        "split": "one style per whole game/camp; source match-level train/validation split retained",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for style in STYLES:
        for role in ROLES:
            source = args.source_root / f"{role}_tactical_buildings"
            rows.append(_write_subset(source, args.out / "datasets" / style / role, style, assignments))
            (args.out / "checkpoints" / style / role).mkdir(parents=True, exist_ok=True)
            (args.out / "reports" / style / role).mkdir(parents=True, exist_ok=True)
            (args.out / "replays" / style / role).mkdir(parents=True, exist_ok=True)
    with (args.out / "dataset_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(args.out / "dataset_manifest.csv")


if __name__ == "__main__":
    main()
