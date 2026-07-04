"""Config serialization round-trips exactly (manifest -> identical config)."""

import json

from bimo_ik.config import config_from_dict, config_hash, config_to_dict
from bimo_ik.config.config_full import get_config as get_full
from bimo_ik.config.config_lars import get_config as get_lars


def test_roundtrip_preserves_hash():
    for cfg in (get_lars(), get_full()):
        # through JSON, as the manifest stores it
        rebuilt = config_from_dict(json.loads(json.dumps(config_to_dict(cfg))))
        assert config_hash(rebuilt) == config_hash(cfg)
        assert rebuilt.cells == cfg.cells
        assert rebuilt.trajectories == cfg.trajectories
