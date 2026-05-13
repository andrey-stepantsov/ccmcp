

def pytest_configure(config):
    config.addinivalue_line("markers", "integration: requires a running Qdrant instance")
    config.addinivalue_line("markers", "slow: downloads ML models, excluded from CI")
