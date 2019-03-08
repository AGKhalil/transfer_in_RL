"""Generates grid-world environments.
"""
import numpy as np
import itertools
import copy
import random
import matplotlib.pyplot as plt
from collections import defaultdict


class GridWorld:

    """Class for grids. Creates intermediate grids between an initial and final
    grid by removing one structural limit at a time. Provides interface
    with grid dynamics and structure.

    Attributes:
        diff_grid (TYPE): grid difference between initial and final grids.
            Highlights structural barriers
        final_grid (TYPE): final grid (highest number of barriers)
        initial_grid (TYPE): initial grid (least number of barriers)
        list_of_maps (TYPE): list of all maps created; initial, intermediate,
            final
        nA (int): number of possible actions
        nS (TYPE): number of possible states
        possible_actions (list): possible actions allowed for a grid cell
        x_lim (int): grid x-length
        x_start (int): grid possible starting x-coordinate
        y_lim (int): grid y-length
        y_start (int): grid possible starting y-coordinate
    """

    def __init__(self, world, canyon=None):
        """Initializes grids from intermediate grid creation and interfaces.
        """
        self.world = world
        self.canyon = canyon
        self.x_lim = self.world
        self.y_lim = self.world
        self.nA = 3
        self.nS = self.x_lim * self.y_lim

        self.possible_actions = [0, 1, 2, 3]

        if self.world == 4:
            self.x_start = 0
            self.y_start = 3
            self.x_final = 3
            self.y_final = 3
            if self.canyon:
                self.initial_grid = {
                    (0, 0): {'actions': (1, 0, 1, 0), 'done': False, 'reward': -0.1},
                    (0, 1): {'actions': (1, 1, 1, 0), 'done': False, 'reward': -0.1},
                    (0, 2): {'actions': (0, 1, 1, 0), 'done': False, 'reward': -0.1},
                    (0, 3): {'actions': (0, 0, 1, 0), 'done': False, 'reward': -0.1},

                    (1, 0): {'actions': (1, 0, 1, 1), 'done': False, 'reward': -0.1},
                    (1, 1): {'actions': (1, 1, 0, 1), 'done': False, 'reward': -0.1},
                    (1, 2): {'actions': (1, 1, 1, 1), 'done': False, 'reward': -0.1},
                    (1, 3): {'actions': (0, 1, 1, 1), 'done': False, 'reward': -0.1},

                    (2, 0): {'actions': (0, 0, 1, 1), 'done': False, 'reward': -0.1},
                    (2, 1): {'actions': (1, 0, 0, 0), 'done': False, 'reward': -0.1},
                    (2, 2): {'actions': (0, 1, 1, 1), 'done': False, 'reward': -0.1},
                    (2, 3): {'actions': (0, 0, 0, 1), 'done': False, 'reward': -0.1},

                    (3, 0): {'actions': (1, 0, 0, 1), 'done': False, 'reward': -0.1},
                    (3, 1): {'actions': (1, 1, 0, 0), 'done': False, 'reward': -0.1},
                    (3, 2): {'actions': (1, 1, 0, 1), 'done': False, 'reward': -0.1},
                    (3, 3): {'actions': (0, 1, 0, 0), 'done': True, 'reward': 1},
                }
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
            else:
                self.initial_grid = {
                    (0, 0): {'actions': (1, 0, 1, 0), 'done': False, 'reward': -0.1},
                    (0, 1): {'actions': (1, 1, 1, 0), 'done': False, 'reward': -0.1},
                    (0, 2): {'actions': (0, 1, 1, 0), 'done': False, 'reward': -0.1},
                    (0, 3): {'actions': (0, 0, 1, 0), 'done': False, 'reward': -0.1},

                    (1, 0): {'actions': (1, 0, 1, 1), 'done': False, 'reward': -0.1},
                    (1, 1): {'actions': (1, 1, 0, 1), 'done': False, 'reward': -0.1},
                    (1, 2): {'actions': (1, 1, 1, 1), 'done': False, 'reward': -0.1},
                    (1, 3): {'actions': (0, 1, 1, 1), 'done': False, 'reward': -0.1},

                    (2, 0): {'actions': (0, 0, 1, 1), 'done': False, 'reward': -0.1},
                    (2, 1): {'actions': (1, 0, 0, 0), 'done': False, 'reward': -0.1},
                    (2, 2): {'actions': (1, 1, 0, 1), 'done': False, 'reward': -0.1},
                    (2, 3): {'actions': (0, 1, 0, 1), 'done': False, 'reward': -0.1},

                    (3, 0): {'actions': (1, 0, 0, 1), 'done': False, 'reward': -0.1},
                    (3, 1): {'actions': (1, 1, 0, 0), 'done': False, 'reward': -0.1},
                    (3, 2): {'actions': (1, 1, 0, 0), 'done': False, 'reward': -0.1},
                    (3, 3): {'actions': (0, 1, 0, 0), 'done': True, 'reward': 1},
                }
                self.final_grid = {
                    (0, 0): {'actions': (1, 0, 1, 0), 'done': False, 'reward': -0.1},
                    (0, 1): {'actions': (1, 1, 0, 0), 'done': False, 'reward': -0.1},
                    (0, 2): {'actions': (0, 1, 1, 0), 'done': False, 'reward': -0.1},
                    (0, 3): {'actions': (0, 0, 1, 0), 'done': False, 'reward': -0.1},

                    (1, 0): {'actions': (0, 0, 1, 1), 'done': False, 'reward': -0.1},
                    (1, 1): {'actions': (0, 0, 0, 0), 'done': False, 'reward': -0.1},
                    (1, 2): {'actions': (1, 0, 0, 1), 'done': False, 'reward': -0.1},
                    (1, 3): {'actions': (0, 1, 1, 1), 'done': False, 'reward': -0.1},

                    (2, 0): {'actions': (0, 0, 1, 1), 'done': False, 'reward': -0.1},
                    (2, 1): {'actions': (1, 0, 0, 0), 'done': False, 'reward': -0.1},
                    (2, 2): {'actions': (1, 1, 0, 0), 'done': False, 'reward': -0.1},
                    (2, 3): {'actions': (0, 1, 0, 1), 'done': False, 'reward': -0.1},

                    (3, 0): {'actions': (1, 0, 0, 1), 'done': False, 'reward': -0.1},
                    (3, 1): {'actions': (1, 1, 0, 0), 'done': False, 'reward': -0.1},
                    (3, 2): {'actions': (1, 1, 0, 0), 'done': False, 'reward': -0.1},
                    (3, 3): {'actions': (0, 1, 0, 0), 'done': True, 'reward': 1},
                }
        elif self.world == 9:
            self.x_start = 0
            self.y_start = 5
            self.x_final = 5
            self.y_final = 4
            self.initial_grid = {
                (0, 0): {'actions': (0, 0, 0, 0), 'done': False, 'reward': -0.1},
                (0, 1): {'actions': (0, 0, 0, 0), 'done': False, 'reward': -0.1},
                (0, 2): {'actions': (0, 0, 0, 0), 'done': False, 'reward': -0.1},
                (0, 3): {'actions': (1, 0, 1, 0), 'done': False, 'reward': -0.1},
                (0, 4): {'actions': (1, 1, 1, 0), 'done': False, 'reward': -0.1},
                (0, 5): {'actions': (1, 1, 1, 0), 'done': False, 'reward': -0.1},
                (0, 6): {'actions': (0, 1, 1, 0), 'done': False, 'reward': -0.1},
                (0, 7): {'actions': (0, 0, 0, 0), 'done': False, 'reward': -0.1},
                (0, 8): {'actions': (0, 0, 0, 0), 'done': False, 'reward': -0.1},

                (1, 0): {'actions': (0, 0, 1, 0), 'done': False, 'reward': -0.1},
                (1, 1): {'actions': (0, 0, 0, 0), 'done': False, 'reward': -0.1},
                (1, 2): {'actions': (1, 0, 1, 0), 'done': False, 'reward': -0.1},
                (1, 3): {'actions': (1, 1, 1, 1), 'done': False, 'reward': -0.1},
                (1, 4): {'actions': (1, 1, 1, 1), 'done': False, 'reward': -0.1},
                (1, 5): {'actions': (1, 1, 1, 1), 'done': False, 'reward': -0.1},
                (1, 6): {'actions': (1, 1, 1, 1), 'done': False, 'reward': -0.1},
                (1, 7): {'actions': (0, 1, 1, 0), 'done': False, 'reward': -0.1},
                (1, 8): {'actions': (0, 0, 0, 0), 'done': False, 'reward': -0.1},

                (2, 0): {'actions': (1, 0, 1, 1), 'done': False, 'reward': -0.1},
                (2, 1): {'actions': (1, 1, 1, 0), 'done': False, 'reward': -0.1},
                (2, 2): {'actions': (1, 1, 1, 1), 'done': False, 'reward': -0.1},
                (2, 3): {'actions': (1, 1, 1, 1), 'done': False, 'reward': -0.1},
                (2, 4): {'actions': (1, 1, 1, 1), 'done': False, 'reward': -0.1},
                (2, 5): {'actions': (1, 1, 1, 1), 'done': False, 'reward': -0.1},
                (2, 6): {'actions': (1, 1, 1, 1), 'done': False, 'reward': -0.1},
                (2, 7): {'actions': (1, 1, 1, 1), 'done': False, 'reward': -0.1},
                (2, 8): {'actions': (0, 0, 0, 0), 'done': False, 'reward': -0.1},

                (3, 0): {'actions': (1, 0, 1, 1), 'done': False, 'reward': -0.1},
                (3, 1): {'actions': (1, 1, 1, 1), 'done': False, 'reward': -0.1},
                (3, 2): {'actions': (1, 1, 1, 1), 'done': False, 'reward': -0.1},
                (3, 3): {'actions': (1, 1, 1, 1), 'done': False, 'reward': -0.1},
                (3, 4): {'actions': (1, 1, 1, 1), 'done': False, 'reward': -0.1},
                (3, 5): {'actions': (1, 1, 1, 1), 'done': False, 'reward': -0.1},
                (3, 6): {'actions': (1, 1, 1, 1), 'done': False, 'reward': -0.1},
                (3, 7): {'actions': (1, 1, 1, 1), 'done': False, 'reward': -0.1},
                (3, 8): {'actions': (0, 1, 1, 1), 'done': False, 'reward': -0.1},

                (4, 0): {'actions': (1, 0, 1, 1), 'done': False, 'reward': -0.1},
                (4, 1): {'actions': (1, 1, 1, 1), 'done': False, 'reward': -0.1},
                (4, 2): {'actions': (1, 1, 0, 1), 'done': False, 'reward': -0.1},
                (4, 3): {'actions': (1, 1, 1, 1), 'done': False, 'reward': -0.1},
                (4, 4): {'actions': (1, 1, 1, 1), 'done': False, 'reward': -0.1},
                (4, 5): {'actions': (1, 1, 1, 1), 'done': False, 'reward': -0.1},
                (4, 6): {'actions': (1, 1, 1, 1), 'done': False, 'reward': -0.1},
                (4, 7): {'actions': (1, 1, 1, 1), 'done': False, 'reward': -0.1},
                (4, 8): {'actions': (0, 1, 0, 1), 'done': False, 'reward': -0.1},

                (5, 0): {'actions': (1, 0, 1, 1), 'done': False, 'reward': -0.1},
                (5, 1): {'actions': (0, 1, 0, 1), 'done': False, 'reward': -0.1},
                (5, 2): {'actions': (0, 0, 0, 0), 'done': False, 'reward': -0.1},
                (5, 3): {'actions': (1, 0, 1, 1), 'done': False, 'reward': -0.1},
                (5, 4): {'actions': (1, 1, 1, 1), 'done': True, 'reward': 1},
                (5, 5): {'actions': (1, 1, 1, 1), 'done': False, 'reward': -0.1},
                (5, 6): {'actions': (1, 1, 1, 1), 'done': False, 'reward': -0.1},
                (5, 7): {'actions': (0, 1, 0, 1), 'done': False, 'reward': -0.1},
                (5, 8): {'actions': (0, 0, 0, 0), 'done': False, 'reward': -0.1},

                (6, 0): {'actions': (0, 0, 1, 1), 'done': False, 'reward': -0.1},
                (6, 1): {'actions': (0, 0, 0, 0), 'done': False, 'reward': -0.1},
                (6, 2): {'actions': (0, 0, 0, 0), 'done': False, 'reward': -0.1},
                (6, 3): {'actions': (1, 0, 1, 1), 'done': False, 'reward': -0.1},
                (6, 4): {'actions': (1, 1, 1, 1), 'done': False, 'reward': -0.1},
                (6, 5): {'actions': (1, 1, 0, 1), 'done': False, 'reward': -0.1},
                (6, 6): {'actions': (0, 1, 1, 1), 'done': False, 'reward': -0.1},
                (6, 7): {'actions': (0, 0, 0, 0), 'done': False, 'reward': -0.1},
                (6, 8): {'actions': (0, 0, 0, 0), 'done': False, 'reward': -0.1},

                (7, 0): {'actions': (1, 0, 1, 1), 'done': False, 'reward': -0.1},
                (7, 1): {'actions': (0, 1, 1, 0), 'done': False, 'reward': -0.1},
                (7, 2): {'actions': (0, 0, 0, 0), 'done': False, 'reward': -0.1},
                (7, 3): {'actions': (1, 0, 1, 1), 'done': False, 'reward': -0.1},
                (7, 4): {'actions': (0, 1, 0, 1), 'done': False, 'reward': -0.1},
                (7, 5): {'actions': (0, 0, 0, 0), 'done': False, 'reward': -0.1},
                (7, 6): {'actions': (1, 0, 0, 1), 'done': False, 'reward': -0.1},
                (7, 7): {'actions': (0, 1, 1, 0), 'done': False, 'reward': -0.1},
                (7, 8): {'actions': (0, 0, 0, 0), 'done': False, 'reward': -0.1},

                (8, 0): {'actions': (1, 0, 0, 1), 'done': False, 'reward': -0.1},
                (8, 1): {'actions': (1, 1, 0, 1), 'done': False, 'reward': -0.1},
                (8, 2): {'actions': (1, 1, 0, 0), 'done': False, 'reward': -0.1},
                (8, 3): {'actions': (0, 1, 0, 1), 'done': False, 'reward': -0.1},
                (8, 4): {'actions': (0, 0, 0, 0), 'done': False, 'reward': -0.1},
                (8, 5): {'actions': (0, 0, 0, 0), 'done': False, 'reward': -0.1},
                (8, 6): {'actions': (0, 0, 0, 0), 'done': False, 'reward': -0.1},
                (8, 7): {'actions': (1, 0, 0, 1), 'done': False, 'reward': -0.1},
                (8, 8): {'actions': (0, 1, 0, 0), 'done': False, 'reward': -0.1},
            }
            self.final_grid = {
                (0, 0): {'actions': (0, 0, 0, 0), 'done': False, 'reward': -0.1},
                (0, 1): {'actions': (0, 0, 0, 0), 'done': False, 'reward': -0.1},
                (0, 2): {'actions': (0, 0, 0, 0), 'done': False, 'reward': -0.1},
                (0, 3): {'actions': (1, 0, 1, 0), 'done': False, 'reward': -0.1},
                (0, 4): {'actions': (1, 1, 1, 0), 'done': False, 'reward': -0.1},
                (0, 5): {'actions': (1, 1, 1, 0), 'done': False, 'reward': -0.1},
                (0, 6): {'actions': (0, 1, 1, 0), 'done': False, 'reward': -0.1},
                (0, 7): {'actions': (0, 0, 0, 0), 'done': False, 'reward': -0.1},
                (0, 8): {'actions': (0, 0, 0, 0), 'done': False, 'reward': -0.1},

                (1, 0): {'actions': (0, 0, 1, 0), 'done': False, 'reward': -0.1},
                (1, 1): {'actions': (0, 0, 0, 0), 'done': False, 'reward': -0.1},
                (1, 2): {'actions': (1, 0, 1, 0), 'done': False, 'reward': -0.1},
                (1, 3): {'actions': (1, 1, 1, 1), 'done': False, 'reward': -0.1},
                (1, 4): {'actions': (1, 1, 0, 1), 'done': False, 'reward': -0.1},
                (1, 5): {'actions': (1, 1, 1, 1), 'done': False, 'reward': -0.1},
                (1, 6): {'actions': (1, 1, 1, 1), 'done': False, 'reward': -0.1},
                (1, 7): {'actions': (0, 1, 1, 0), 'done': False, 'reward': -0.1},
                (1, 8): {'actions': (0, 0, 0, 0), 'done': False, 'reward': -0.1},

                (2, 0): {'actions': (1, 0, 1, 1), 'done': False, 'reward': -0.1},
                (2, 1): {'actions': (1, 1, 1, 0), 'done': False, 'reward': -0.1},
                (2, 2): {'actions': (1, 1, 1, 1), 'done': False, 'reward': -0.1},
                (2, 3): {'actions': (0, 1, 0, 1), 'done': False, 'reward': -0.1},
                (2, 4): {'actions': (0, 0, 0, 0), 'done': False, 'reward': -0.1},
                (2, 5): {'actions': (1, 0, 0, 1), 'done': False, 'reward': -0.1},
                (2, 6): {'actions': (1, 1, 1, 1), 'done': False, 'reward': -0.1},
                (2, 7): {'actions': (1, 1, 1, 1), 'done': False, 'reward': -0.1},
                (2, 8): {'actions': (0, 0, 0, 0), 'done': False, 'reward': -0.1},

                (3, 0): {'actions': (1, 0, 1, 1), 'done': False, 'reward': -0.1},
                (3, 1): {'actions': (1, 1, 1, 1), 'done': False, 'reward': -0.1},
                (3, 2): {'actions': (0, 1, 1, 1), 'done': False, 'reward': -0.1},
                (3, 3): {'actions': (0, 0, 0, 0), 'done': False, 'reward': -0.1},
                (3, 4): {'actions': (0, 0, 0, 0), 'done': False, 'reward': -0.1},
                (3, 5): {'actions': (0, 0, 0, 0), 'done': False, 'reward': -0.1},
                (3, 6): {'actions': (1, 0, 1, 1), 'done': False, 'reward': -0.1},
                (3, 7): {'actions': (1, 1, 1, 1), 'done': False, 'reward': -0.1},
                (3, 8): {'actions': (0, 1, 1, 1), 'done': False, 'reward': -0.1},

                (4, 0): {'actions': (1, 0, 1, 1), 'done': False, 'reward': -0.1},
                (4, 1): {'actions': (1, 1, 1, 1), 'done': False, 'reward': -0.1},
                (4, 2): {'actions': (1, 1, 0, 1), 'done': False, 'reward': -0.1},
                (4, 3): {'actions': (1, 1, 1, 0), 'done': False, 'reward': -0.1},
                (4, 4): {'actions': (1, 1, 1, 0), 'done': False, 'reward': -0.1},
                (4, 5): {'actions': (1, 1, 1, 0), 'done': False, 'reward': -0.1},
                (4, 6): {'actions': (1, 1, 1, 1), 'done': False, 'reward': -0.1},
                (4, 7): {'actions': (1, 1, 1, 1), 'done': False, 'reward': -0.1},
                (4, 8): {'actions': (0, 1, 0, 1), 'done': False, 'reward': -0.1},

                (5, 0): {'actions': (1, 0, 1, 1), 'done': False, 'reward': -0.1},
                (5, 1): {'actions': (0, 1, 0, 1), 'done': False, 'reward': -0.1},
                (5, 2): {'actions': (0, 0, 0, 0), 'done': False, 'reward': -0.1},
                (5, 3): {'actions': (1, 0, 1, 1), 'done': False, 'reward': -0.1},
                (5, 4): {'actions': (1, 1, 1, 1), 'done': True, 'reward': 1},
                (5, 5): {'actions': (1, 1, 1, 1), 'done': False, 'reward': -0.1},
                (5, 6): {'actions': (1, 1, 1, 1), 'done': False, 'reward': -0.1},
                (5, 7): {'actions': (0, 1, 0, 1), 'done': False, 'reward': -0.1},
                (5, 8): {'actions': (0, 0, 0, 0), 'done': False, 'reward': -0.1},

                (6, 0): {'actions': (0, 0, 1, 1), 'done': False, 'reward': -0.1},
                (6, 1): {'actions': (0, 0, 0, 0), 'done': False, 'reward': -0.1},
                (6, 2): {'actions': (0, 0, 0, 0), 'done': False, 'reward': -0.1},
                (6, 3): {'actions': (1, 0, 1, 1), 'done': False, 'reward': -0.1},
                (6, 4): {'actions': (1, 1, 1, 1), 'done': False, 'reward': -0.1},
                (6, 5): {'actions': (1, 1, 0, 1), 'done': False, 'reward': -0.1},
                (6, 6): {'actions': (0, 1, 1, 1), 'done': False, 'reward': -0.1},
                (6, 7): {'actions': (0, 0, 0, 0), 'done': False, 'reward': -0.1},
                (6, 8): {'actions': (0, 0, 0, 0), 'done': False, 'reward': -0.1},

                (7, 0): {'actions': (1, 0, 1, 1), 'done': False, 'reward': -0.1},
                (7, 1): {'actions': (0, 1, 1, 0), 'done': False, 'reward': -0.1},
                (7, 2): {'actions': (0, 0, 0, 0), 'done': False, 'reward': -0.1},
                (7, 3): {'actions': (1, 0, 1, 1), 'done': False, 'reward': -0.1},
                (7, 4): {'actions': (0, 1, 0, 1), 'done': False, 'reward': -0.1},
                (7, 5): {'actions': (0, 0, 0, 0), 'done': False, 'reward': -0.1},
                (7, 6): {'actions': (1, 0, 0, 1), 'done': False, 'reward': -0.1},
                (7, 7): {'actions': (0, 1, 1, 0), 'done': False, 'reward': -0.1},
                (7, 8): {'actions': (0, 0, 0, 0), 'done': False, 'reward': -0.1},

                (8, 0): {'actions': (1, 0, 0, 1), 'done': False, 'reward': -0.1},
                (8, 1): {'actions': (1, 1, 0, 1), 'done': False, 'reward': -0.1},
                (8, 2): {'actions': (1, 1, 0, 0), 'done': False, 'reward': -0.1},
                (8, 3): {'actions': (0, 1, 0, 1), 'done': False, 'reward': -0.1},
                (8, 4): {'actions': (0, 0, 0, 0), 'done': False, 'reward': -0.1},
                (8, 5): {'actions': (0, 0, 0, 0), 'done': False, 'reward': -0.1},
                (8, 6): {'actions': (0, 0, 0, 0), 'done': False, 'reward': -0.1},
                (8, 7): {'actions': (1, 0, 0, 1), 'done': False, 'reward': -0.1},
                (8, 8): {'actions': (0, 1, 0, 0), 'done': False, 'reward': -0.1},
            }

        self.reset_state()

        self.list_of_maps = [copy.deepcopy(self.final_grid)]

        self.diff_grid = copy.deepcopy(self.final_grid)

    def show_grid(self, c_map):
        """Print a given grid to terminal.

        Args:
            c_map (TYPE): a given grid
        """
        for y in range(self.y_lim - 1, -1, -1):
            print('-' * self.x_lim * 9)
            for x in range(self.x_lim):
                if x == self.x_final and y == self.y_final:
                    direct = ''.join(self.arrow(4))
                else:
                    direct = ''.join(self.arrow(i, j)
                                     for i, j in enumerate(
                        c_map[(x, y)]['actions']))
                print('{:^7}'.
                      format(direct.replace(" ", "")), "|", end="")
            print("")
        print('-' * self.x_lim * 9)

    def show_policy(self, c_map):
        """Print policy to terminal to a grid.

        Args:
            c_map (TYPE): a given policy
        """
        for y in range(self.y_lim - 1, -1, -1):
            print('-' * self.x_lim * 9)
            for x in range(self.x_lim):
                if x == self.x_final and y == self.y_final:
                    direct = ''.join(self.arrow(4))
                else:
                    direct = ''.join(self.arrow(c_map.get((x, y), None)))
                print('{:^7}'.format(direct.replace(" ", "")), "|", end="")
            print("")
        print('-' * self.x_lim * 9)

    def arrow(self, direction, val=1):
        """Converts an action to an arrow for printing to terminal.

        Args:
            direction (TYPE): direction of action
            val (int, optional): if action is possible for a cell

        Returns:
            TYPE: direction representation for printing
        """
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
        """Creates differential grid to highlight structural barriers between
        the initial and final grids.
        """
        for key, value in self.final_grid.items():
            self.diff_grid[key]['actions'] = np.subtract(
                self.initial_grid[key]['actions'], value['actions'])

    def make_maps(self):
        """Creates intermediate grids by removing a single barrier one at
        a time. Can randomize the process to create different grid
        trajectories.
        """
        self.make_diff_grid()
        new_map = copy.deepcopy(self.final_grid)
        other_key = ()

        if self.world == 4:
            barrier_cells = []
            for key, value in self.diff_grid.items():
                # print(value)
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
                        self.update_grid(new_map, new_map, key,
                                         other_key, look, other_look)
                        self.list_of_maps.append(copy.deepcopy(new_map))
        elif self.world == 9:
            barrier_cells = []
            for key, value in self.diff_grid.items():
                if sum(value['actions']) == 4:
                    barrier_cells.append(key)
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

                    # if other_key:
                    if sum(self.diff_grid[other_key]['actions']) != 4:
                        self.update_grid(new_map, new_map, key,
                                         other_key, look, other_look)
                self.list_of_maps.append(copy.deepcopy(new_map))

    def update_grid(self, c_map, new_map, key, other_key, look, other_look):
        """Updates the new map with the barrier removal change. Also removes
        barriers from self.diff_grid for the next update so it is not repeated.

        Args:
            c_map (TYPE): current map
            new_map (TYPE): updated map
            key (TYPE): key of cell to be changed
            other_key (TYPE): key of the adjacent cell to also be changed
            look (TYPE): action direction. represents location of barrier
                with respect to cell
            other_look (TYPE): action direction of adjacent cell to be changed.
        """
        new_map[key]['actions'] = np.add(
            c_map[key]['actions'], np.eye(1, self.nA + 1, look)[0])
        new_map[other_key]['actions'] = np.add(
            new_map[other_key]['actions'],
            np.eye(1, self.nA + 1, other_look)[0])
        self.diff_grid[key]['actions'] = np.subtract(
            self.diff_grid[key]['actions'], np.eye(1, self.nA + 1, look)[0])
        self.diff_grid[other_key]['actions'] = np.subtract(
            self.diff_grid[other_key]['actions'],
            np.eye(1, self.nA + 1, other_look)[0])

    def reset_state(self):
        """Randomly chooses initial state on grid between four possible cells.
        (0, 3), (0, 2), (1, 3), (1, 2)

        Returns:
            TYPE: Initial state
        """
        p = np.random.random()
        if p < 0.25:
            return (self.x_start, self.y_start)
        elif p < 0.5:
            return (self.x_start + 1, self.y_start)
        elif p < 0.75:
            return (self.x_start, self.y_start - 1)
        elif p < 1:
            return (self.x_start + 1, self.y_start - 1)

if __name__ == "__main__":
    grid = GridWorld(9)
    # grid.make_diff_grid()
    # grid.show_grid(grid.diff_grid)
    grid.make_maps()
    grid.list_of_maps.reverse()
    for mymap in grid.list_of_maps:
        grid.show_grid(mymap)

    print(len(grid.list_of_maps))
