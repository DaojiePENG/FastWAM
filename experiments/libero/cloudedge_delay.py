from collections import deque


class ObservationDelayHistory:
    def __init__(self, max_delay_steps: int):
        if max_delay_steps < 0:
            raise ValueError("max_delay_steps must be non-negative")
        self.max_delay_steps = int(max_delay_steps)
        self._items = deque(maxlen=self.max_delay_steps + 1)

    def append(self, observation):
        self._items.append(observation)

    @property
    def current(self):
        if not self._items:
            raise RuntimeError("delay history is empty")
        return self._items[-1]

    def sample(self, rng):
        if self.max_delay_steps == 0 or len(self._items) < 2:
            return self.current, 0
        maximum = min(self.max_delay_steps, len(self._items) - 1)
        delay = int(rng.randint(1, maximum + 1))
        return self._items[-1 - delay], delay
