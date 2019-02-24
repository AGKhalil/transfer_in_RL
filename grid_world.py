import numpy as np
import itertools
import copy
import random
import matplotlib.pyplot as plt
from collections import defaultdict

class GridWorld:

    def __init__(self):

        self.x_lim = 4
        self.y_lim = 4
        self.nA = 3
        self.nS = self.x_lim * self.y_lim

        # Start position
        self.x_start = 0
        self.y_start = 3

        self.reset_state()

        # [U, D, R, L]
        self.possible_actions = [0, 1, 2, 3]

        # (U, D, R, L, T)
        self.initial_grid = {
            (0, 0): {'actions': (1, 0, 1, 0), 'done': False, 'reward': -0.1},
            (0, 1): {'actions': (1, 1, 1, 0), 'done': False, 'reward': -0.1},
            (0, 2): {'actions': (0, 1, 1, 0), 'done': False, 'reward': -0.1},
            (0, 3): {'actions': (0, 0, 1, 0), 'done': False, 'reward': -0.1},
            (1, 0): {'actions': (1, 0, 1, 1), 'done': False, 'reward': -0.1},
            (1, 1): {'actions': (0, 1, 0, 1), 'done': False, 'reward': -0.1},
            (1, 2): {'actions': (1, 0, 1, 1), 'done': False, 'reward': -0.1},
            (1, 3): {'actions': (0, 1, 1, 1), 'done': False, 'reward': -0.1},
            (2, 0): {'actions': (0, 0, 1, 1), 'done': False, 'reward': -0.1},
            (2, 1): {'actions': (1, 0, 1, 0), 'done': False, 'reward': -0.1},
            (2, 2): {'actions': (0, 1, 1, 1), 'done': False, 'reward': -0.1},
            (2, 3): {'actions': (0, 0, 0, 1), 'done': False, 'reward': -0.1},
            (3, 0): {'actions': (1, 0, 0, 1), 'done': False, 'reward': -0.1},
            (3, 1): {'actions': (1, 1, 0, 1), 'done': False, 'reward': -0.1},
            (3, 2): {'actions': (1, 1, 0, 1), 'done': False, 'reward': -0.1},
            (3, 3): {'actions': (0, 1, 0, 0), 'done': True, 'reward': 1},
        }

        # (U, D, R, L, T)
        self.final_grid = {
            (0, 0): {'actions': (1, 0, 1, 0), 'done': False, 'reward': -0.1},
            (0, 1): {'actions': (1, 1, 0, 0), 'done': False, 'reward': -0.1},
            (0, 2): {'actions': (0, 1, 1, 0), 'done': False, 'reward': -0.1},
            (0, 3): {'actions': (0, 0, 1, 0), 'done': False, 'reward': -0.1},
            (1, 0): {'actions': (1, 0, 1, 1), 'done': False, 'reward': -0.1},
            (1, 1): {'actions': (0, 1, 0, 0), 'done': False, 'reward': -0.1},
            (1, 2): {'actions': (1, 0, 0, 1), 'done': False, 'reward': -0.1},
            (1, 3): {'actions': (0, 1, 1, 1), 'done': False, 'reward': -0.1},
            (2, 0): {'actions': (0, 0, 1, 1), 'done': False, 'reward': -0.1},
            (2, 1): {'actions': (1, 0, 0, 0), 'done': False, 'reward': -0.1},
            (2, 2): {'actions': (0, 1, 0, 0), 'done': False, 'reward': -0.1},
            (2, 3): {'actions': (0, 0, 0, 1), 'done': False, 'reward': -0.1},
            (3, 0): {'actions': (1, 0, 0, 1), 'done': False, 'reward': -0.1},
            (3, 1): {'actions': (1, 1, 0, 0), 'done': False, 'reward': -0.1},
            (3, 2): {'actions': (1, 1, 0, 0), 'done': False, 'reward': -0.1},
            (3, 3): {'actions': (0, 1, 0, 0), 'done': True, 'reward': 1},
        }

        self.list_of_maps = [copy.deepcopy(self.final_grid)]

        # self.states = self.all_states()
        self.diff_grid = copy.deepcopy(self.final_grid)
        self.diff_grid = self.make_diff_grid()

    def show_grid(self, c_map):
        for  y in range(self.y_lim-1, -1, -1):
            print('-'*self.x_lim*9)
            for x in range(self.x_lim):
                direct = ''.join(self.arrow(i, j) for i, j in enumerate(c_map[(x, y)]['actions']))
                print('{:^7}'.format(direct.replace(" ", "")), "|", end="")
            print("")
        print('-'*self.x_lim*9)

    def show_policy(self, c_map):
        for  y in range(self.y_lim-1, -1, -1):
            print('-'*self.x_lim*9)
            for x in range(self.x_lim):
                if x == self.x_lim - 1 and y == self.y_lim - 1:
                    direct = ''.join(self.arrow(4))
                else:
                    direct = ''.join(self.arrow(c_map.get((x, y), None)))
                print('{:^7}'.format(direct.replace(" ", "")), "|", end="")
            print("")
        print('-'*self.x_lim*9)

    def arrow(self, direction, val=1):
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
            elif not direction:
                return ' '
        else:
            return ' '

    def make_diff_grid(self):
        for key, value in self.diff_grid.items():
            self.diff_grid[key]['actions'] = np.subtract(self.initial_grid[key]['actions'], value['actions'])
        return self.diff_grid

    def make_maps(self):
        new_map = copy.deepcopy(self.final_grid)
        other_key = ()
        barrier_cells = []
        for key, value in self.diff_grid.items():
            if 1 in value['actions']:
                barrier_cells.append(key)

        # randomizes the barrier cells
        # for i, j in enumerate(barrier_cells):
        #     swap = random.randrange(i, len(barrier_cells))
        #     barrier_cells[i], barrier_cells[swap] = barrier_cells[swap], j

        for key in barrier_cells:
            value = self.diff_grid[key]['actions']
            for look in np.nonzero(value)[0]:
                x, y = key[0], key[1]
                if look == 0:
                    if self.diff_grid[(x, y + 1)]['actions'][1]:
                        other_key = (x, y + 1)
                        other_look = 1
                elif look == 1:
                    if self.diff_grid[(x, y - 1)]['actions'][0]:
                        other_key = (x, y - 1)
                        other_look = 0
                elif look == 2:
                    if self.diff_grid[(x + 1, y)]['actions'][3]:
                        other_key = (x + 1, y)
                        other_look = 3
                elif look == 3:
                    if self.diff_grid[(x - 1, y)]['actions'][2]:
                        other_key = (x - 1, y)
                        other_look = 2
                if other_key:
                    self.update_grid(new_map, new_map, key, other_key, look, other_look)
                    self.list_of_maps.append(copy.deepcopy(new_map))

    def update_grid(self, c_map, new_map, key, other_key, look, other_look):
        new_map[key]['actions'] = np.add(c_map[key]['actions'], np.eye(1, self.nA + 1, look)[0])
        new_map[key]['actions'] = np.add(c_map[key]['actions'], self.diff_grid[key]['actions'])
        new_map[other_key]['actions'] = np.add(new_map[other_key]['actions'], np.eye(1, self.nA + 1, other_look)[0])
        self.diff_grid[key]['actions'] = np.subtract(self.diff_grid[key]['actions'], np.eye(1, self.nA + 1, look)[0])
        self.diff_grid[other_key]['actions'] = np.subtract(self.diff_grid[other_key]['actions'], np.eye(1, self.nA + 1, other_look)[0])

    def reset_state(self):
        p = np.random.random()
        if p < 0.25:
            return (self.x_start, self.y_start)
        elif p < 0.5:
            return (self.x_start + 1, self.y_start)
        elif p < 0.75:
            return (self.x_start, self.y_start - 1)
        elif p < 1:
            return (self.x_start + 1, self.y_start - 1)

# if __name__ == "__main__":

    # grid = GridWorld()
    # print("Initial Grid")
    # grid.show_grid(grid.initial_grid)
    # print("Final Grid")
    # grid.show_grid(grid.final_grid)
    # print("Differential Grid")
    # grid.show_grid(grid.make_diff_grid())
    # print("Multiple Grids In-between")
    # grid.initial_grid
    # grid.make_maps()
    # for c_map in grid.list_of_maps:
    #     grid.show_grid(c_map)

    # print(len(grid.list_of_maps))