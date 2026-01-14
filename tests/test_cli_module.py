import runpy
import pytest


def test_cli_module_runs_and_exits():
    # Running the package as a module without args should raise SystemExit
    # (argparse will call parser.error when no --input or --gui is provided)
    with pytest.raises(SystemExit):
        runpy.run_module('sagevision.cli', run_name='__main__')
