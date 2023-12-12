import sys
from builtins import *
from os.path import dirname as up

import pytest
import torch
from dantro._import_tools import import_module_from_path
from pkg_resources import resource_filename

from utopya.yaml import load_yml

sys.path.insert(0, up(up(up(__file__))))

utils = import_module_from_path(mod_path=up(up(up(__file__))), mod_str="include.utils")

# Load the test config
CFG_FILENAME = resource_filename("tests", "cfgs/test_utils.yml")
test_cfg = load_yml(CFG_FILENAME)


def test_random_tensor():
    def _test_entry(cfg, tensor):
        if cfg["distribution"] == "uniform":
            assert cfg["parameters"]["lower"] <= tensor <= cfg["parameters"]["upper"]

    for _, config in test_cfg.items():
        _raises = config.pop("_raises", False)
        _exp_exc = Exception if not isinstance(_raises, str) else globals()[_raises]
        _warns = config.pop("_warns", False)
        _exp_warning = UserWarning if not isinstance(_warns, str) else globals()[_warns]
        _match = config.pop("_match", " ")

        cfg = config if "cfg" not in config.keys() else config.get("cfg")

        if not _raises:
            for size in [(1,), (4, 4, 4)]:
                t = utils.random_tensor(cfg, size=size)

                if isinstance(cfg, list):
                    assert len(t) == len(cfg)
                else:
                    assert t.shape == torch.Size(size)

                t = torch.flatten(t)
                for _ in range(len(t)):
                    if isinstance(cfg, dict):
                        _test_entry(cfg, t[_])
                    else:
                        _test_entry(cfg[_], t[_])

        if not _raises and not _warns:
            utils.random_tensor(cfg, size=(1,))

        elif _warns and not _raises:
            with pytest.warns(_exp_warning, match=_match):
                utils.random_tensor(cfg, size=(1,))

        elif _raises and not _warns:
            with pytest.raises(_exp_exc, match=_match):
                utils.random_tensor(cfg, size=(1,))
