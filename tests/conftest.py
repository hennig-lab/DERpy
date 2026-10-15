import os

import pytest


def pytest_addoption(parser):
    parser.addoption("--example-data-dir", default=os.getenv("DER_EXAMPLE_DATA_DIR"))
    parser.addoption("--der-data-dir", default=os.getenv("DER_FULL_DATA_DIR"))
    parser.addoption("--der-reference-dir", action="append", default=None)
    parser.addoption("--der-report", default="der-validation.json")
    parser.addoption("--der-min-agreement", type=float, default=None)
    parser.addoption("--der-detailed-report", action="store_true", default=False)


@pytest.fixture
def example_data_dir(request):
    path = request.config.getoption("--example-data-dir")
    if not path:
        pytest.skip("Supply --example-data-dir for the two uploaded examples")
    return path
