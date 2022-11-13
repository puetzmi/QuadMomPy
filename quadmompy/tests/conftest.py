import pytest

# Automatically change working directory to current test directory
@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)

# Global seed for random number generators
def pytest_configure():
    pytest.random_seed = 125125
