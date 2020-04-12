import numpy as np

class TurtleAgent(object):
    LEARNING_RATE = 0.2
    DISCOUNT = 0.8
    EXPLORATION_RATE = 0.05
    DISCRETE_STATE_SIZE = [25, 25, 16]
    MAX_INDICES = [i - 1 for i in DISCRETE_STATE_SIZE]

    def __init__(self, observation_space, action_space):
        self._observation_space = observation_space
        self._action_space = action_space
        self._discrete_space_window_size = (observation_space.high - observation_space.low) / self.DISCRETE_STATE_SIZE
        self._q_table = np.random.uniform(low = 0, high=1, size=(self.DISCRETE_STATE_SIZE + [action_space.n]))

    def _to_discrete_state(self, state):
        normalized_state = (state - self._observation_space.low) / self.DISCRETE_STATE_SIZE
        normalized_state = np.minimum(normalized_state, self.MAX_INDICES)
        return tuple(normalized_state.astype(np.int))

    def act(self, state):
        if np.random.uniform() < self.EXPLORATION_RATE:
            return self._action_space.sample()
        discrete_state = self._to_discrete_state(state)
        action_array = self._q_table[discrete_state]
        return np.argmax(action_array)

    def feedback(self, state, action, new_state, reward):
        discrete_state = self._to_discrete_state(state)
        discrete_new_state = self._to_discrete_state(new_state)

        current_q = self._q_table[discrete_state + (action,)]
        max_future_q = np.max(self._q_table[discrete_new_state])
        new_q = (1 - self.LEARNING_RATE) * current_q + \
            self.LEARNING_RATE * (reward + self.DISCOUNT * max_future_q)
        self._q_table[discrete_state + (action,)] = new_q
