import numpy as np
import pytest

from experiments.libero.cloudedge_delay import ObservationDelayHistory


def test_zero_delay_returns_current():
    history = ObservationDelayHistory(0)
    history.append("now")
    assert history.sample(np.random.RandomState(0)) == ("now", 0)


def test_delay_never_exceeds_available_history():
    history = ObservationDelayHistory(20)
    history.append(0)
    history.append(1)
    assert history.sample(np.random.RandomState(0)) == (0, 1)


def test_invalid_delay_rejected():
    with pytest.raises(ValueError):
        ObservationDelayHistory(-1)
