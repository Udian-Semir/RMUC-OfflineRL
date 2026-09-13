from pathlib import Path

from setuptools import find_packages, setup


package_name = "simulation2d"


def data_files(directory: str) -> list[tuple[str, list[str]]]:
    files = [str(path) for path in Path(directory).iterdir() if path.is_file()]
    return [(f"share/{package_name}/{directory}", files)] if files else []


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml", "README.md"]),
        (f"share/{package_name}/launch", [str(path) for path in Path("launch").glob("*.py")]),
        *data_files("checkpoints"),
        *data_files("assets"),
        *data_files("config"),
    ],
    install_requires=["setuptools"],
    zip_safe=False,
    maintainer="julia",
    maintainer_email="julia@todo.todo",
    description="Interactive 2D tactical world model and radar decision tools.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "simulation2d_smoke = simulation2d.tools.smoke:main",
            "simulation2d_node = simulation2d.runtime.live_node:main",
            "simulation2d_visualizer = simulation2d.visualization.live_visualizer:main",
            "simulation2d_costmap_smoke = simulation2d.tools.costmap_smoke:main",
            "simulation2d_train = simulation2d.policy.train_cli:main",
            "simulation2d_replay = simulation2d.tools.replay_interactive:main",
            "simulation2d_evaluate = simulation2d.tools.evaluate_ppo_checkpoints:main",
            "simulation2d_map_preview = simulation2d.tools.foxglove_semantic_preview:main",
            "simulation2d_validate_map = simulation2d.tools.validate_semantic_map:main",
        ],
    },
)
