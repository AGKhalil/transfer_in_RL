"""Applies SARSA algorithm to grid-world environments.
"""
import sys
import numpy as np
import itertools
import copy
import random
import matplotlib.pyplot as plt
from collections import defaultdict
from grid_world import GridWorld


class SARSA:

    """Creates value and policy containers for SARSA evaluation of a
    grid-world instance.

    Attributes:
        alpha (float): learning rate
        c_map (TYPE): grid-world instance
        discount_factor (float): discount factor
        epsilon (float): policy choice probability
        policy (TYPE): action policy
        possible_actions (TYPE): possible actions for an agent to take
        Q (TYPE): state action value storage
        x_lim (TYPE): grid x-length
        y_lim (TYPE): grid y-length
    """

    def __init__(self, c_map, possible_actions, x_lim, y_lim, Q=None):
        """Initializes SARSA parameters

        Args:
            c_map (TYPE): grid-world instance
            possible_actions (TYPE): possible actions for an agent to take
            x_lim (TYPE): grid x-length
            y_lim (TYPE): grid y-length
        """
        self.alpha = 0.5
        self.discount_factor = 1
        self.epsilon = 0

        self.c_map = c_map
        self.possible_actions = possible_actions
        self.x_lim = x_lim
        self.y_lim = y_lim

        if Q is None:
            Q = dict()
        self.Q = Q
        self.policy = dict()

    def epsilon_greedy_random_action(self, state):
        """Chooses action based on a greedy policy, but allows for
        exploration by a 1 - epsilon probability.

        Args:
            state (TYPE): current state

        Returns:
            TYPE: action to be taken
        """
        p = np.random.random()
        count = 0
        if p < self.epsilon:
            action = random.choice(self.possible_actions)
        else:
            # print(self.Q)
            q_all = [self.Q.get((state, a), 0.0)
                     for a in self.possible_actions]
            max_a = [a for a in self.possible_actions if q_all[
                a] == max(q_all)]
            if len(max_a) > 1:
                action = random.choice(max_a)
            else:
                action = max_a[0]
                # print("ACTION", action)
        self.policy[state] = action
        return action

    def update_Q(self, state, action, new_state, new_action, reward):
        """Updates state value function

        Args:
            state (TYPE): current state
            action (TYPE): action taken from current state
            new_state (TYPE): new state
            new_action (TYPE): action taken from new state
            reward (TYPE): reward received upon taking an action from
                the current state
        """
        # if self.Q.get((state, action), None):
        # print("Q before is", self.Q.get((state, action), 0.0))
        q = self.Q.get((state, action), 0.0)
        self.Q[state, action] = q + self.alpha * \
            (reward + self.discount_factor *
             self.Q.get((new_state, action), 0.0) - q)
        # print("Q after is", self.Q[state, action])
        # print("state", state)
        # else:
        #     self.Q[state, action] = reward

    def take_step(self, state, action):
        """Agent performs action to transition to next state on grid-world.

        Args:
            state (TYPE): current state
            action (TYPE): action taken from current state

        Returns:
            TYPE: new state and reward received from said state
        """
        # from action [U, D, R, L] to state
        x, y = state[0], state[1]
        if self.c_map[state]['actions'][action]:
            if action == 0:
                new_state = (x, y + 1)
            if action == 1:
                new_state = (x, y - 1)
            if action == 2:
                new_state = (x + 1, y)
            if action == 3:
                new_state = (x - 1, y)
        else:
            new_state = state
        return new_state, self.c_map[new_state]['reward']

    def print_values(self):
        """Print state value function to terminal
        """
        for y in range(self.y_lim - 1, -1, -1):
            print('-' * self.x_lim * 9)
            for x in range(self.x_lim):
                value = np.mean([self.Q.get(((x, y), a), 0.0)
                             for a in self.possible_actions])
                if value >= 0:
                    print(' {:.4f}'.format(value), "|", end="")
                else:
                    print('{:.4f}'.format(value), "|", end="")
            print("")
        print('-' * self.x_lim * 9)
