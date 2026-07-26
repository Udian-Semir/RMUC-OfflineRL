"""Live reward/cost telemetry for single-sentry PPO training.

The logger always writes a CSV and the latest PNG.  ``--live`` additionally
opens an interactive Matplotlib window when a graphical backend is available;
on a headless machine it degrades to the same periodically refreshed PNG.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping


class TrainingDashboard:
    """Persist metrics and optionally render a four-panel training dashboard."""

    def __init__(self, out_dir: str | Path, *, live: bool = False) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.out_dir / "metrics.csv"
        self.png_path = self.out_dir / "training_live.png"
        self._csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
        self._writer: csv.DictWriter | None = None
        self.rows: list[dict[str, float]] = []
        self.live = bool(live)
        self._plt = None
        self._fig = None
        self._axes = None
        self._interactive = False
        if self.live:
            try:
                import matplotlib.pyplot as plt
            except ImportError as exc:  # pragma: no cover - depends on optional local GUI stack
                self.close()
                raise RuntimeError("--live requires matplotlib; install it with `pip install matplotlib`") from exc
            self._plt = plt
            self._interactive = "agg" not in str(plt.get_backend()).lower()
            plt.ion()
            self._fig, self._axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
            self._fig.suptitle("Sentry PPO training telemetry")
            if self._interactive:
                self._fig.show()

    def update(self, update: int, metrics: Mapping[str, float]) -> None:
        row = {"update": float(update)}
        row.update({key: float(value) for key, value in metrics.items()})
        self.rows.append(row)
        if self._writer is None:
            self._writer = csv.DictWriter(self._csv_file, fieldnames=list(row), extrasaction="ignore")
            self._writer.writeheader()
        self._writer.writerow(row)
        self._csv_file.flush()
        self._render()

    def _render(self) -> None:
        if not self.rows:
            return
        if self._plt is None or self._axes is None:
            return
        x = [row["update"] for row in self.rows]

        def series(key: str) -> list[float]:
            return [row.get(key, float("nan")) for row in self.rows]

        axes = self._axes.ravel()
        for axis in axes:
            axis.clear()
        axes[0].plot(x, series("mean_episode_return"), label="episode return")
        axes[0].plot(x, series("mean_reward"), label="rollout reward")
        axes[0].set_title("Reward")
        axes[0].set_xlabel("PPO update")
        axes[0].legend(loc="best")

        axes[1].plot(x, series("mean_path_cost"), label="path cost")
        axes[1].plot(x, series("mean_path_risk"), label="path risk")
        axes[1].set_title("Navigation cost")
        axes[1].set_xlabel("PPO update")
        axes[1].legend(loc="best")

        axes[2].plot(x, series("mean_damage_dealt"), label="damage dealt")
        axes[2].plot(x, series("mean_damage_taken"), label="damage taken")
        axes[2].plot(x, series("mean_blue_outpost_damage"), label="blue outpost")
        axes[2].plot(x, series("mean_red_outpost_damage"), label="red outpost loss")
        axes[2].plot(x, series("mean_red_outpost_control_loss"), label="red control loss")
        axes[2].set_title("Combat and objective terms")
        axes[2].set_xlabel("PPO update")
        axes[2].legend(loc="best")

        axes[3].plot(x, series("mean_invalid_action"), label="invalid action")
        axes[3].plot(x, series("mean_goal_switch"), label="goal switch")
        axes[3].plot(x, series("mean_goal_switch_blocked"), label="switch blocked")
        axes[3].plot(x, series("entropy"), label="policy entropy")
        axes[3].set_title("Constraint and exploration signals")
        axes[3].set_xlabel("PPO update")
        axes[3].legend(loc="best")

        self._fig.savefig(self.png_path, dpi=130)
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        if self._interactive:
            self._plt.pause(0.001)

    def close(self) -> None:
        if getattr(self, "_csv_file", None) is not None and not self._csv_file.closed:
            self._csv_file.close()
        if self._plt is not None and self._fig is not None:
            self._fig.savefig(self.png_path, dpi=130)
            self._plt.ioff()
            self._plt.close(self._fig)

    def __enter__(self) -> "TrainingDashboard":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
