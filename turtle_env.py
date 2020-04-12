"""
gym environment for Turtle tasks.
"""
import numpy as np
import gym
from gym import core, spaces
import turtle
import math
import abc


class BaseTurtleEnv(gym.Env, metaclass=abc.ABCMeta):
    """
    Observation: (location_x, location_y, orientation)
    """
    MAX_LOCATION_X = 400
    MAX_LOCATION_Y = 400

    def __init__(self):
        self._possible_actions = self._get_possible_actions()
        self._turtle : turtle.Turtle = turtle.Turtle()
        self._turtle.speed(0)
        self.action_space = spaces.Discrete(len(self._possible_actions))
        high = np.array([self.MAX_LOCATION_X, self.MAX_LOCATION_Y, 360], dtype=np.float32)
        low = np.array([-self.MAX_LOCATION_X, -self.MAX_LOCATION_Y, 0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.reset()

    def enable_draw(self):
        turtle.tracer(1)

    def disable_draw(self):
        turtle.tracer(False)

    def step(self, action):
        action_mathod = self._possible_actions[action]
        action_mathod()
        
        position_x, position_y = self._turtle.pos()
        angle = self._turtle.heading()
        state = (position_x, position_y, angle)

        reward, done, info = self._evaluate_current_state(action, state)

        return state, reward, done, info

    def reset(self):
        self._turtle.reset()

    def render(self, mode='human'):
        pass #  TODO

    def save_to_file(self, filename):
        self._turtle.getscreen().getcanvas().postscript(file=filename)

    @abc.abstractmethod
    def _get_possible_actions(self):
        raise NotImplemented()

    @abc.abstractmethod
    def _evaluate_current_state(self, action, state):
        raise NotImplemented()


class StayAwayFromCenterEnvironment(BaseTurtleEnv):
    """
    The challange: Stay as far as possible from (0, 0), without falling from the border
    """
    def _get_possible_actions(self):
        def forward():
            self._turtle.forward(50)
        def right():
            self._turtle.circle(10, 90)
        return [forward, right]

    def _evaluate_current_state(self, action, state):
        position_x, position_y, angle = state
        done = False
        is_out_of_bound = abs(position_x) > self.MAX_LOCATION_X or abs(position_y) > self.MAX_LOCATION_Y

        distance_from_origin = self._turtle.distance(0, 0)
        reward = distance_from_origin / 10
        if is_out_of_bound:
            done = True
            reward -= 1000
        if action == 0: # Move forward bonus
            reward += 10
        info = {}
        return reward, done, info


class DontTouchTheWallEnvironment(BaseTurtleEnv):
    """
    The challange: Stay as long as you can without falling from the border
    """
    def _get_possible_actions(self):
        def right():
            self._turtle.circle(100, 20)
        def left():
            self._turtle.circle(-100, 20)
        return [right, left]

    def _evaluate_current_state(self, action, state):
        position_x, position_y, angle = state
        done = False
        is_out_of_bound = abs(position_x) > self.MAX_LOCATION_X or abs(position_y) > self.MAX_LOCATION_Y

        if is_out_of_bound:
            done = True
            reward = -100
        else:
            reward = 10
        info = {}
        return reward, done, info
