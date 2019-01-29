# python -m pytest -v

from lin_reg_sgd.linreg import LinearRegression

import pytest


@pytest.fixture
def lr():
    return LinearRegression()


def test_type(lr):
    assert isinstance(lr, LinearRegression)
