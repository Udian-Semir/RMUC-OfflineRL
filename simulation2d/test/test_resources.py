import yaml

from simulation2d.resources import asset_path, checkpoint_path, config_path


def test_packaged_resources_exist() -> None:
    assert asset_path("semantic_map_aligned.json").is_file()
    assert asset_path("blackwhite_map.png").is_file()
    assert asset_path("blackwhite_astar_inflated_0p35m.png").is_file()
    assert config_path("demo.yaml").is_file()
    assert checkpoint_path().is_file()


def test_live_config_matches_checkpoint_opponent_mode() -> None:
    with config_path("live_ppo_00155_offline.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    assert config["env"]["sparring_backend"] == "offline"
    assert config["env"]["sparring_update_seconds"] == 1.0
    assert config["env"]["sparring_fire_policy"] == "intent_dps"
