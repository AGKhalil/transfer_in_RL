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
    returns = 0
    list_returns = []

    print("Currently at task ", task)
    for episode in range(nEp):
        episode_return = 0
        state = grid.reset_state()
        action = sarsa.epsilon_greedy_random_action(state)
        for step in itertools.count():
            new_state, reward = sarsa.take_step(state, action)
            returns += reward
            episode_return += reward
            new_action = sarsa.epsilon_greedy_random_action(new_state)
            sarsa.update_Q(state, action, new_state, new_action, reward)

            # print("Episode", episode, "Step", step, "Return", episode_return)

            if sarsa.c_map[new_state]['done']:
                steps += step
                list_returns.append(episode_return)
                break
            else:
                state, action = new_state, new_action

        current_mean = abs(np.mean(list(np.sum(sarsa.Q.values()))))
        if np.abs(old_mean - current_mean) < delta:
            # list_steps.append(steps)
            # list_returns.append(returns)
            print("Convergence at episode ", episode)
            print("Total of steps ", steps)
            print("Total return", returns)
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

    return sarsa.Q, list_returns, episode

if __name__ == "__main__":

    print("-" * 100)

    # Evaluation
    returns = 0
    steps = 0
    all_returns = []
    all_episodes = []

    # create grid-world instance
    canyon = True
    grid = GridWorld(not canyon)
    grid.make_maps()

    possible_actions = grid.possible_actions
    x_lim, y_lim = grid.x_lim, grid.y_lim
    grid.list_of_maps.reverse()

    Q = None
    for task, current_map in enumerate(grid.list_of_maps, 1):
        print("-" * 50)
        # creates SARSA instance
        sarsa = SARSA(current_map, possible_actions, x_lim, y_lim, Q)
        Q, returns, episodes = do_task(sarsa, grid, task)
        all_returns.append(returns)
        all_episodes.append(episodes)
    print("-" * 100)

    flat_returns = [item for sublist in all_returns for item in sublist]
    avg_returns = [np.mean(flat_returns[:i])
                   for i in range(1, len(flat_returns))]
    val = 0
    for i in all_episodes:
        val += i
        plt.axvline(x=val, linestyle='--')
    plt.plot(flat_returns, label="Immediate Return")
    plt.plot(avg_returns, label="Averaged Return")
    plt.xlabel("Number of Episodes")
    plt.ylabel("Return")
    plt.legend(loc="lower right")
    plt.show()
