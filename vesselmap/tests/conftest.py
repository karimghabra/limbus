def pytest_configure(config):
    config.addinivalue_line("markers", "slow: runs the full mapper (a minute or more)")
