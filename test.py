import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict

GAMMA = 0.9

class GridWorld:

    def __init__(self):

        self.x_lim = 4
        self.y_lim = 4
        self.act_lim = 3

        # Start position
        self.x_start = 0
        self.y_start = 0

        # self.reset_state()

        self.rewards = {
            (0, 0): -0.1,
            (0, 1): -0.1,
            (0, 2): -0.1,
            (0, 3): -0.1,
            (1, 0): -0.1,
            (1, 1): -0.1,
            (1, 2): -0.1,
            (1, 3): -0.1,
            (2, 0): -0.1,
            (2, 1): -0.1,
            (2, 2): -0.1,
            (2, 3): -0.1,
            (3, 0): -0.1,
            (3, 1): -0.1,
            (3, 2): -0.1,
            (3, 3): 1,
        }

        # (U, D, R, L, T)
        self.initial_grid = {
            (0, 0): (1, 0, 1, 0),
            (0, 1): (1, 1, 1, 0),
            (0, 2): (0, 1, 1, 0),
            (0, 3): (0, 0, 1, 0),
            (1, 0): (1, 0, 1, 1),
            (1, 1): (0, 1, 0, 1),
            (1, 2): (1, 0, 1, 1),
            (1, 3): (0, 1, 1, 1),
            (2, 0): (0, 0, 1, 1),
            (2, 1): (1, 0, 1, 0),
            (2, 2): (0, 1, 1, 1),
            (2, 3): (0, 0, 0, 1),
            (3, 0): (1, 0, 0, 1),
            (3, 1): (1, 1, 0, 1),
            (3, 2): (1, 1, 0, 1),
            (3, 3): (0, 1, 0, 0),
        }

        # (U, D, R, L, T)
        self.final_grid = {
            (0, 0): (1, 0, 1, 0),
            (0, 1): (1, 1, 0, 0),
            (0, 2): (0, 1, 1, 0),
            (0, 3): (0, 0, 1, 0),
            (1, 0): (1, 0, 1, 1),
            (1, 1): (0, 1, 0, 0),
            (1, 2): (1, 0, 0, 1),
            (1, 3): (0, 1, 1, 1),
            (2, 0): (0, 0, 1, 1),
            (2, 1): (1, 0, 0, 0),
            (2, 2): (0, 1, 0, 0),
            (2, 3): (0, 0, 0, 1),
            (3, 0): (1, 0, 0, 1),
            (3, 1): (1, 1, 0, 0),
            (3, 2): (1, 1, 0, 0),
            (3, 3): (0, 1, 0, 0),
        }

        self.values = {
            (0, 0): 0,
            (0, 1): 0,
            (0, 2): 0,
            (0, 3): 0,
            (1, 0): 0,
            (1, 1): 0,
            (1, 2): 0,
            (1, 3): 0,
            (2, 0): 0,
            (2, 1): 0,
            (2, 2): 0,
            (2, 3): 0,
            (3, 0): 0,
            (3, 1): 0,
            (3, 2): 0,
            (3, 3): 0,
        }

        self.policy = {
            (0, 0): '',
            (0, 1): '',
            (0, 2): '',
            (1, 0): '',
            (1, 1): '',
            (1, 2): '',
            (2, 0): '',
            (2, 1): '',
            (2, 2): '',
        }

        # self.states = self.all_states()
        self.diff_grid = {**self.final_grid}

    def show_grid(self, map):
        for  y in range(self.y_lim-1, -1, -1):
            print('-'*self.x_lim*9)
            for x in range(self.x_lim):
                direct = ''.join(self.arrow(i, j) for i, j in enumerate(map.get((x, y))))
                print('{:^7}'.format(direct.replace(" ", "")), "|", end="")
            print("")
        print('-'*self.x_lim*9)

    def arrow(self, direction, val):
        # print(val)
        if val:
            if direction == 0:
                return '\u2191'
            elif direction == 1:
                return '\u2193'
            elif direction == 2:
                return '\u2192'
            elif direction == 3:
                return '\u2190'
            elif direction == 4:
                return 'X'
        else:
            return ' '

    def make_diff_grid(self):
        for key, value in self.final_grid.items():
            self.diff_grid[key] = np.subtract(self.initial_grid[key], value)
        return self.diff_grid

    def make_maps(self):
        new_map = {**self.final_grid}
        other_key = ()
        barrier_cells = []
        for key, value in self.diff_grid.items():
            if 1 in value:
                barrier_cells.append(key)

        # randomizes the barrier cells
        # for i, j in enumerate(barrier_cells):
        #     swap = random.randrange(i, len(barrier_cells))
        #     barrier_cells[i], barrier_cells[swap] = barrier_cells[swap], j

        for key in barrier_cells:
            value = self.diff_grid[key]
            for look in np.nonzero(value)[0]:
                # look = np.nonzero(value)[0][0]
                # print("I am look ", look)
                x, y = key[0], key[1]
                if look == 0:
                    if self.diff_grid.get((x, y + 1))[1]:
                        other_key = (x, y + 1)
                        other_look = 1
                elif look == 1:
                    if self.diff_grid.get((x, y - 1))[0]:
                        other_key = (x, y - 1)
                        other_look = 0
                elif look == 2:
                    if self.diff_grid.get((x + 1, y))[3]:
                        other_key = (x + 1, y)
                        other_look = 3
                elif look == 3:
                    # print("K diff ", self.diff_grid.get((x, y)))
                    # print("OK diff ", self.diff_grid.get((x - 1, y)))
                    # print("X and Y ", x, " ", y)
                    if self.diff_grid.get((x - 1, y))[2]:
                        other_key = (x - 1, y)
                        # print("I get here")
                        other_look = 2
                if other_key:
                    # print("Look and Other Look ", look, " ", other_look)
                    self.update_grid(new_map, new_map, key, other_key, look, other_look)
                    self.show_grid(new_map)
        return new_map

    def update_grid(self, map, new_map, key, other_key, look, other_look):
        # print("Before key ", new_map[key])
        new_map[key] = np.add(map[key], np.eye(1, self.act_lim + 1, look)[0])
        # print("After key ", new_map[key])
        new_map[key] = np.add(map[key], self.diff_grid[key])
        # print("Before other_key ", new_map[other_key])
        # print("Look", np.eye(1, self.act_lim + 1, look)[0])
        # print("Other look", np.eye(1, self.act_lim + 1, other_look)[0])
        new_map[other_key] = np.add(new_map[other_key], np.eye(1, self.act_lim + 1, other_look)[0])
        # print("After other_key ", new_map[other_key])
        self.diff_grid[key] = np.subtract(self.diff_grid[key], np.eye(1, self.act_lim + 1, look)[0])
        self.diff_grid[other_key] = np.subtract(self.diff_grid[other_key], np.eye(1, self.act_lim + 1, other_look)[0])

    # def reset_state(self):
    #     self.x = self.x_start
    #     self.y = self.y_start

    # def set_random_policy(self):
    #     self.policy = {}
    #     for state in self.actions.keys():
    #         # action_index = np.random.random_integers(len(self.actions[state])) - 1
    #         # self.policy[state] = self.actions[state][action_index]
    #         self.policy[state] = self.get_random_action(state)

    # def get_random_action(self, state):
    #     action_index = np.random.random_integers(len(self.actions[state])) - 1
    #     action_to_return = self.actions[state][action_index]
    #     return action_to_return

    # def epsilon_greedy_random_action(self, state, epsilon=0.1):
    #     p = np.random.random()
    #     if p < epsilon:
    #         action = self.get_random_action(state)
    #     else:
    #         action = self.policy[state]
    #     return action

    # def reset_values(self):
    #     self.values = {
    #         (0, 0): 0,
    #         (1, 0): 0,
    #         (0, 1): 0,
    #         (1, 1): 0,
    #         (0, 2): 0,
    #         (1, 2): 0,
    #     }

    # def set_state(self, state):
    #     self.x = state[0]
    #     self.y = state[1]

    # def current_state(self):
    #     return (self.x, self.y)

    # def all_states(self):
    #     return set(self.actions.keys()) | set(self.rewards.keys())

    # def is_terminal(self, state):
    #     return state not in self.actions

    # def step(self, action):
    #     if action in self.actions[self.current_state()]:
    #         if action == 'R':
    #             self.x += 1
    #         elif action == 'L':
    #             self.x -= 1
    #         elif action == 'U':
    #             self.y += 1
    #         elif action == 'D':
    #             self.y -= 1
    #     return self.current_state(), self.rewards.get(self.current_state(), 0), self.game_over(self.current_state())
 
    # def undo_step(self, action):
    #     if action == 'R':
    #         self.x -= 1
    #     elif action == 'L':
    #         self.x += 1
    #     elif action == 'U':
    #         self.y -= 1
    #     elif action == 'D':
    #         self.y += 1
    #     assert(self.current_state() in self.all_states())

    # def game_over(self, state):
    #     return state not in self.actions

    # def print_policy(self):
    #     for y in range(self.y_lim-1, -1, -1):
    #         print("------------")
    #         for x in range(self.x_lim):
    #             action = self.policy.get((x, y), " ")
    #             print("  %s  |" % action, end="")
    #         print("")
    #     print("------------")

    # def print_values(self):
    #     for  y in range(self.y_lim-1, -1, -1):
    #         print("------------")
    #         for x in range(self.x_lim):
    #             value = self.values.get((x, y), 0)
    #             if value >= 0:
    #                 print(" %.2f|" % value, end="")
    #             else:
    #                 print("%.2f|" % value, end="")
    #         print("")
    #     print("------------")

    # def print_rewards(self):
    #     for  y in range(self.y_lim-1, -1, -1):
    #         print("------------")
    #         for x in range(self.x_lim):
    #             reward = self.rewards.get((x, y), 0)
    #             if reward >= 0:
    #                 print(" %.2f|" % reward, end="")
    #             else:
    #                 print("%.2f|" % reward, end="")
    #         print("")
    #     print("------------")

    # def max_dict(self, dict_in):
    #     max_key = None
    #     max_value = float('-inf')
    #     for key, value in dict_in.items():
    #         if value > max_value:
    #             max_value = value
    #             max_key = key
    #     return max_key, max_value

    # def play_game(self):
    #     num_steps = 0

    #     # Start state
    #     self.reset_state()
    #     done = False
    #     states_actions_rewards = []

    #     while (not done) and (num_steps < 5):
    #         old_state = self.current_state()
    #         action = self.epsilon_greedy_random_action(self.current_state())
    #         state, reward, done = self.step(action)
    #         num_steps = num_steps + 1
    #         states_actions_rewards.append((old_state, action, reward))
    #         # print(old_state, action, reward)
    #     # print("Episode end")
    #     # print(states_actions_rewards)

    #     G = 0
    #     states_actions_returns = []

    #     for state, action, reward in reversed(states_actions_rewards):
    #         print(state, action, reward)
    #         G = reward + GAMMA * G
    #         states_actions_returns.append((state, action, G))
            
    #     states_actions_returns.reverse()

    #     return states_actions_returns


if __name__ == "__main__":

    num_ep = 25

    grid = GridWorld()
    print("Initial Grid")
    grid.show_grid(grid.initial_grid)
    print("Final Grid")
    grid.show_grid(grid.final_grid)
    print("Differential Grid")
    grid.show_grid(grid.make_diff_grid())
    print("Multiple Grids In-between")
    grid.make_maps()
