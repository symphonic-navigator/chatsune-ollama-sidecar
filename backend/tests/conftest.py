import logging
import sys
from pathlib import Path

import pytest
import structlog

# Make `sidecar` importable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True, scope="session")
def _configure_structlog_for_stdlib():
    """Route structlog through stdlib logging so pytest's caplog can see it.

    Tests that want JSON output (test_logging.py) call configure_logging()
    themselves, which overrides this wiring. We don't touch root handlers
    here — caplog installs its own handler and must keep control of root.
    """
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.KeyValueRenderer(
                key_order=["event"], drop_missing=True
            ),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )
    yield
