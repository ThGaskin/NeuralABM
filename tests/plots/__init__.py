import os

def _str2bool(val: str):
    """Copy of strtobool from deprecated distutils package"""
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif val in ("n", "no", "f", "false", "off", "0"):
        return False
    raise ValueError(f"Invalid truth value {repr(val)}!")


# .. Test-related variables ...................................................

USE_TEST_OUTPUT_DIR: bool = _str2bool(
    os.environ.get("USE_TEST_OUTPUT_DIR", "false")
)
"""Whether to use the test output directory. Can be set via the environment
variable ``USE_TEST_OUTPUT_DIR``.

NOTE It depends on the tests if they actually take this flag into account!
     If using the ``tmpdir_or_local_dir`` fixture, this will be used to decide
     whether to return a tmpdir or a path within ``TEST_OUTPUT_DIR``.
"""

ABBREVIATE_TEST_OUTPUT_DIR: bool = _str2bool(
    os.environ.get("ABBREVIATE_TEST_OUTPUT_DIR", "false")
)

TEST_OUTPUT_DIR: str = os.path.join(os.path.dirname(__file__), "_output")
"""An output directory that *can* be used to locally store data, e.g. for
looking at plot output. By default, this will be in the ``tests`` directory
itself, but if the `TEST_OUTPUT_DIR`` environment variable is set, will
use that path instead.
"""