"""
Pytest configuration and fixtures for YourDaddy AI Assistant tests.

This file contains shared fixtures and configuration for all tests.
"""

import pytest
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def project_root_path():
    """Return the project root directory path."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_dir(project_root_path):
    """Return the test data directory path."""
    data_dir = project_root_path / "tests" / "test_data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture(scope="function")
def temp_db(tmp_path):
    """Create a temporary database for testing."""
    db_path = tmp_path / "test.db"
    yield db_path
    # Cleanup
    if db_path.exists():
        db_path.unlink()


@pytest.fixture(scope="function")
def mock_env_vars(monkeypatch):
    """Set up mock environment variables for testing."""
    test_vars = {
        'DEBUG': 'true',
        'TESTING': 'true',
    }
    for key, value in test_vars.items():
        monkeypatch.setenv(key, value)
    return test_vars


@pytest.fixture(scope="session")
def skip_if_no_api_key():
    """Skip test if API keys are not configured."""
    def _skip_if_no_key(key_name):
        if not os.getenv(key_name):
            pytest.skip(f"Skipping test: {key_name} not configured")
    return _skip_if_no_key


@pytest.fixture(scope="function")
def clean_test_databases(project_root_path):
    """Clean up test databases before and after tests."""
    test_dbs = [
        project_root_path / "test_memory.db",
        project_root_path / "test_chat_history.db",
    ]
    
    # Cleanup before test
    for db in test_dbs:
        if db.exists():
            db.unlink()
    
    yield
    
    # Cleanup after test
    for db in test_dbs:
        if db.exists():
            db.unlink()


# Pytest hooks
def pytest_configure(config):
    """Configure pytest with custom settings."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow-running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add unit marker to tests in unit/ directory
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Add integration marker to tests in integration/ directory
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
