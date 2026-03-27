"""
config.py — Environment-specific configuration.

ENV is read from the ENV environment variable.
  dev  → liberal thresholds, verbose logging, small dataset
  stg  → production-like settings but isolated DB/storage
  prod → strict thresholds, minimal logging, full dataset
"""

import os

ENV = os.getenv("ENV", "dev")  # default to dev


BASE_CONFIG = {
    "model_path": "models/",
    "random_state": 42,
    "test_size": 0.2,
}

DEV_CONFIG = {
    **BASE_CONFIG,
    "min_accuracy_threshold": 0.80,   # lenient — we're experimenting
    "max_iter": 100,
    "log_level": "DEBUG",
    "description": "Development — fast iteration, verbose output",
}

STG_CONFIG = {
    **BASE_CONFIG,
    "min_accuracy_threshold": 0.90,   # tighter — validates the release
    "max_iter": 200,
    "log_level": "INFO",
    "description": "Staging — production mirror for integration tests",
}

PROD_CONFIG = {
    **BASE_CONFIG,
    "min_accuracy_threshold": 0.93,   # strict — only best models ship
    "max_iter": 500,
    "log_level": "WARNING",
    "description": "Production — real traffic, strict quality gates",
}

_CONFIGS = {
    "dev": DEV_CONFIG,
    "stg": STG_CONFIG,
    "prod": PROD_CONFIG,
}


def get_config():
    cfg = _CONFIGS.get(ENV, DEV_CONFIG)
    print(f"⚙️  Loaded config for ENV={ENV}: {cfg['description']}")
    return cfg
