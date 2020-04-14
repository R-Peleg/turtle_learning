import gym
import numpy as np
import random
from keras import Sequential, optimizers
from keras.layers import InputLayer, Dense, Dropout


class NeuralNetworkAgent(object):
    EPSILON = 0.5
    EPSILON_MULT = 0.98
    MIN_EPSILON = 0.1
    DISCOUNT = 0.95

    def __init__(self, observation_space : gym.Space, action_space : gym.Space):
        self._epsilon = self.EPSILON
        self._action_space = action_space
        self._observation_space = observation_space
        self._model = Sequential()
        self._model.add(InputLayer(input_shape=observation_space.shape))
        self._model.add(Dense(3, activation='selu'))
        self._model.add(Dense(3, activation='selu'))
        self._model.add(Dense(3, activation='selu'))
        self._model.add(Dense(3, activation='selu'))
        self._model.add(Dense(action_space.n, activation='softmax'))
        optimizer = optimizers.Adamax(learning_rate=0.01)
        self._model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])

    def decay_epsilon(self):
        if self._epsilon > self.MIN_EPSILON:
            self._epsilon *= self.EPSILON_MULT

    def _normalize_input(self, state):
        observation_space_size = self._observation_space.high - self._observation_space.low
        return (state - self._observation_space.low) / observation_space_size

    def act(self, state):
        if np.random.random() < self._epsilon:
            return self._action_space.sample()
        normalized_state = self._normalize_input(state)
        prediction = self._model.predict(normalized_state.reshape(-1, len(normalized_state)))
        return np.argmax(prediction[0])

    def feedback(self, feedbacks):
        # TODO: Batch training
        #np_feedbacks = np.array(feedbacks)
        #np_feedbacks[:,0] = self._normalize_input(np.array(list(np_feedbacks[:,0])))
        #np_feedbacks[:,2] = self._normalize_input(np.array(list(np_feedbacks[:,2])))
        # if len(feedbacks) > 100:
        #     feedbacks = random.sample(feedbacks, int(len(feedbacks) / 10))
        train_input = []
        train_target = []
        for (state, action, new_state, reward) in feedbacks:
            normalized_state = self._normalize_input(state)
            normalized_new_state = self._normalize_input(new_state)
            max_future_q = np.max(self._model.predict(normalized_new_state.reshape(-1, len(new_state)))[0])
            target_q = reward + self.DISCOUNT * max_future_q
            target_vector = self._model.predict(normalized_state.reshape(-1, len(state)))[0]
            target_vector[action] = target_q
            train_input.append(normalized_state)
            train_target.append(target_vector)
        self._model.fit(np.array(train_input), np.array(train_target), epochs=1, batch_size=len(train_input), verbose=0)
