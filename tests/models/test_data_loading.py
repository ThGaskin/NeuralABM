import sys
from os.path import dirname as up

from dantro._import_tools import import_module_from_path
from dantro._yaml import load_yml
from pkg_resources import resource_filename

from utopya.testtools import ModelTest

sys.path.insert(0, up(up(up(__file__))))

SIR = import_module_from_path(mod_path=up(up(up(__file__))), mod_str="models.SIR")
HW = import_module_from_path(
    mod_path=up(up(up(__file__))), mod_str="models.HarrisWilson"
)

# Load the test config
CFG_FILENAME = resource_filename("tests", "cfgs/test_data_loading.yml")
test_cfg = load_yml(CFG_FILENAME)


def test_data_loading():
    for _, config in test_cfg.items():
        # Get the model type
        model_name = config.pop("model")

        mtc = ModelTest(model_name)
        model = mtc.create_run_load(**config)

        assert model[1]

        # Load the previously generated data and run again
        if model in ["HarrisWilson", "SIR"]:
            config["parameter_space"][model_name].update(
                {
                    "Data": {
                        "load_from_dir": model[0]._dirs["run"] + "/data/uni0/data.h5"
                    }
                }
            )
        if model in ["Kuramoto", "HarrisWilsonNW"]:
            for ele in ["network", "eigen_frequencies", "training_data"]:
                config["parameter_space"][model_name].update(
                    {
                        "Data": {
                            "load_from_dir": {
                                ele: model[0]._dirs["run"] + "/data/uni0/data.h5"
                            }
                        }
                    }
                )

        model = mtc.create_run_load(**config)

        assert model[1]

        del model
