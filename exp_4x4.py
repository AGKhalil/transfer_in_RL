"""Script for running experiment.
"""
from grid_world import GridWorld
from sarsa import SARSA
import sys
import numpy as np
import itertools
import matplotlib.pyplot as plt

def do_task(sarsa, grid, task):

    # number of maximum episodes to run
    nEp = 200

    # initialize algorithm parameters
    old_mean = 0
    delta = 0.000001
    steps = 0
    plt_steps = 0
    list_steps = []
    returns = 0
    list_returns = []

    print("Currently at task ", task)
    for episode in range(nEp):
        state = grid.reset_state()
        print("Initial State", state)
        action = sarsa.epsilon_greedy_random_action(state)
        for step in range(1000):
            new_state, reward = sarsa.take_step(state, action)
            returns += reward
            # plt_steps += step
            # print("step ", step)
            # print("reward ", reward, " steps ", step)
            list_returns.append(returns)
            list_steps.append(step)
            new_action = sarsa.epsilon_greedy_random_action(new_state)
            sarsa.update_Q(state, action, new_state, new_action, reward)
            # print("Q script", sarsa.Q[state, action])
            # print("state ", state, " action taken", grid.arrow(action), " new state ", new_state)
            # print("new action", new_action)
            # sarsa.print_values()

            if sarsa.c_map[new_state]['done']:
                steps += step
                break
            else:
                state, action = new_state, new_action

        current_mean = abs(np.mean(list(np.sum(sarsa.Q.values()))))
        if np.abs(old_mean - current_mean) < delta:
            # list_steps.append(steps)
            # list_returns.append(returns)
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

    return sarsa.Q, list_returns, list_steps

if __name__ == "__main__":

    print("-"*100)

    # Evaluation
    returns = 0
    steps = 0
    all_returns = []
    all_steps = []

    # create grid-world instance
    grid = GridWorld()
    grid.make_maps()

    possible_actions = grid.possible_actions
    x_lim, y_lim = grid.x_lim, grid.y_lim
    # grid.list_of_maps.reverse()

    # Train initial policy
    sarsa = SARSA(grid.list_of_maps[0], possible_actions, x_lim, y_lim)
    Q, returns, steps = do_task(sarsa, grid, 0)
    print("-"*50)
    # sarsa = SARSA(grid.list_of_maps[0], possible_actions, x_lim, y_lim, Q)
    # _, returns, steps = do_task(sarsa, grid, 0)
    all_returns.append(returns)
    all_steps.append(steps)
    # for task, current_map in enumerate(grid.list_of_maps[1:], 1):
        # creates SARSA instance
        # sarsa = SARSA(current_map, possible_actions, x_lim, y_lim, Q)
        # Q, returns, steps = do_task(sarsa, grid, task)
        # all_returns.append(returns)
        # all_steps.append(steps)
        # print("-"*50)

    print(len(all_returns))
    print(len(all_steps))

    # flat_returns = [item for sublist in all_returns for item in sublist]
    # flat_steps = [item for sublist in all_steps for item in sublist]
    # inter_steps = zip(semi_steps[1:], semi_steps)
    # print(semi_steps[1:])
    # print(semi_steps)
    # cumal_steps = [i + j for i, j in inter_steps]
    # flat_steps = [semi_steps[0]] + cumal_steps
    # print(flat_steps)

    # print(all_steps[0])
    # print(all_returns[0])
    plt.plot(all_steps, all_returns, 'ro')
    plt.show()
