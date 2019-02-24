"""Script for running experiment.
"""
from grid_world import GridWorld
from sarsa import SARSA
import sys
import numpy as np
import itertools

if __name__ == "__main__":

    # number of maximum episodes to run
    nEp = 200

    # create grid-world instance
    grid = GridWorld()
    grid.make_maps()

    # Change index to get different maps 0-4
    current_map = grid.list_of_maps[0]

    if not current_map:
        print("Map index is out of range.")
        sys.exit()

    possible_actions = grid.possible_actions
    x_lim, y_lim = grid.x_lim, grid.y_lim

    # creates SARSA instance
    sarsa = SARSA(current_map, possible_actions, x_lim, y_lim)

    # initialize algorithm parameters
    old_mean = 0
    delta = 0.000001
    steps = 0

    state = grid.reset_state()
    print("Started at ", state)
    for episode in range(nEp):
        action = sarsa.epsilon_greedy_random_action(state)
        for step in itertools.count():
            new_state, reward = sarsa.take_step(state, action)
            new_action = sarsa.epsilon_greedy_random_action(new_state)
            sarsa.update_Q(state, action, new_state, new_action, reward)

            if sarsa.c_map[new_state]['done']:
                steps += step
                break
            else:
                state, action = new_state, new_action

        current_mean = abs(np.mean(list(np.sum(sarsa.Q.values()))))
        if np.abs(old_mean - current_mean) < delta:
            print("Convergence at episode ", episode)
            print("Number of steps ", steps)
            break
        else:
            old_mean = current_mean

    # print results to terminal
    print("Environment Map")
    grid.show_grid(sarsa.c_map)
    print("Environment Values")
    sarsa.print_values()
    print("Environment Policy")
    grid.show_policy(sarsa.policy)
