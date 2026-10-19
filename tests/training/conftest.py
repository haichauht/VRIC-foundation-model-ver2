from __future__ import annotations

import pytest

from tests.models.conftest import _batch_factory, _model_config


@pytest.fixture
def batch_factory():
    return _batch_factory


@pytest.fixture
def model_config():
    return _model_config
