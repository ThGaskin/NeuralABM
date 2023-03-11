import sys
from os.path import dirname as up
from builtins import *
import pytest

from dantro._import_tools import import_module_from_path
from pkg_resources import resource_filename

from utopya.yaml import load_yml

sys.path.insert(0, up(up(up(__file__))))

utils = import_module_from_path(
    mod_path=up(up(up(__file__))), mod_str="include.utils"
)

# Load the test config
CFG_FILENAME = resource_filename("tests", "cfgs/test_utils.yml")
test_cfg = load_yml(CFG_FILENAME)

def test_random_tensor():
    for _, config in test_cfg.items():

        _raises = config.pop("_raises", False)
        _exp_exc = (
            Exception if not isinstance(_raises, str) else globals()[_raises]
        )
        _warns = config.pop("_warns", False)
        _exp_warning = (
            UserWarning if not isinstance(_warns, str) else globals()[_warns]
        )
        _match = config.pop("_match", ' ')

        if not _raises:

            t = utils.random_tensor(**config, size=(1, ))

            if config.get("distribution") == "uniform":
                assert config.get("parameters").get("lower") <= t <= config.get("parameters").get("upper")

            t = utils.random_tensor(**config, size=(4, 4, 4))

            if config.get("distribution") == "uniform":
                assert all([config.get("parameters").get("lower") <= item <= config.get("parameters").get("upper") for item in t.flatten()])

            assert t.shape == (4, 4, 4)

        if not _raises and not _warns:
            t = utils.random_tensor(**config, size=(1, ))

        elif _warns and not _raises:
            with pytest.warns(_exp_warning, match=_match):
                t = utils.random_tensor(**config, size=(1, ))

        elif _raises and not _warns:
            with pytest.raises(_exp_exc, match=_match):
                t = utils.random_tensor(**config, size=(1, ))
