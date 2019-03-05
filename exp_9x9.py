"""Script for running experiment.
"""
from grid_world import GridWorld
from sarsa import SARSA
import sys
import numpy as np
import random
import itertools
import matplotlib.pyplot as plt
import bottleneck as bn
import csv


def do_task(sarsa, grid, task, exploit=False):

    # number of maximum episodes to run
    nEp = 1000

    # initialize algorithm parameters
    old_mean = 0
    delta = 0.000001
    steps = 0
    plt_steps = 0
    returns = 0
    list_returns = []
    sarsa_saves = []
    list_steps = []

    print("Currently at task ", task)
    for episode in range(1, nEp):
        episode_return = 0
        state = grid.reset_state()
        action = sarsa.epsilon_greedy_random_action(state)
        for step in range(500):
            new_state, reward = sarsa.take_step(state, action)
            returns += reward
            episode_return += reward
            new_action = sarsa.epsilon_greedy_random_action(new_state, step, exploit)
            sarsa.update_Q(state, action, new_state, new_action, reward)
            sarsa_saves.append([state, action, reward, new_state, new_action])

            # if step % 100 == 0:
            # print("Task", task, "Episode", episode, "Step", step, "Return", episode_return, "State", state, "Action", grid.arrow(action))
            # grid.show_policy(sarsa.policy)
            # sarsa.print_values()

            if sarsa.c_map[new_state]['done'] or step == 500:
                steps += step
                list_returns.append(episode_return)
                list_steps.append(steps)
                break
            else:
                state, action = new_state, new_action

        current_mean = abs(np.mean(list(np.sum(sarsa.Q.values()))))
        if np.abs(old_mean - current_mean) < delta or episode == nEp - 1:
            print("Convergence at episode ", episode)
            print("Total of steps ", steps)
            print("Cumulative return", returns)
            # print results to terminal
            # print("Environment Map")
            # grid.show_grid(sarsa.c_map)
            # print("Environment Values")
            # sarsa.print_values()
            # print("Environment Policy")
            # grid.show_policy(sarsa.policy)
            return sarsa.Q, list_returns, episode, list_steps
        else:
            old_mean = current_mean

if __name__ == "__main__":
    my_seed = 39 # 20 for non exploiting
    np.random.seed(my_seed)
    random.seed(my_seed * 2)

    print("-" * 100)

    # Evaluation
    tot_steps = 0
    all_returns = []
    all_steps = []

    notrl_tot_steps = 0
    notrl_returns = []
    notrl_steps = []

    # create grid-world instance
    grid = GridWorld(9)
    grid.make_maps()

    possible_actions = grid.possible_actions
    world = grid.world
    grid.list_of_maps.reverse()

    # Direct learning on final grid
    print("Direct learning on final grid")
    sarsa = SARSA(grid.final_grid, possible_actions, world)
    Q, returns, episodes, steps = do_task(
        sarsa, grid, len(grid.list_of_maps) - 1)
    notrl_returns.append(returns)
    notrl_steps.append(steps)
    notrl_tot_steps += steps[-1]
    print("-" * 80)

    # Incremental transfer learning
    print("Incremental transfer learning")
    Q = None
    for task, current_map in enumerate(grid.list_of_maps):
        print("-" * 50)
        # creates SARSA instance
        exploit = False if task == 0 else True
        sarsa = SARSA(current_map, possible_actions, world, Q)
        Q, returns, episodes, steps = do_task(sarsa, grid, task, exploit)

        with open('test.csv', 'w') as f:
            for key in Q.keys():
                f.write("%s,%s\n" % (key, Q[key]))

        all_returns.append(returns)
        tot_counter = 0
        if task != 0:
            tot_counter += all_steps[task - 1][-1]
            all_steps.append([i + tot_counter for i in steps])
        else:
            all_steps.append([i for i in steps])
    print("-" * 100)

    print("Incremental Transfer Cumulative total of steps",
          all_steps[-1][-1] - all_steps[0][-1])
    print("Direct Cumulative total of steps", notrl_steps[-1][-1])

    pre_avg_returns = [bn.move_mean(
        sublist, window=min(len(i) for i in all_returns), min_count=1) for sublist in all_returns]
    avg_returns = [item for sublist in pre_avg_returns for item in sublist]
    flat_returns = [item for sublist in all_returns for item in sublist]
    flat_steps = [item for sublist in all_steps for item in sublist]

    notrl_pre_avg_returns = [bn.move_mean(
        sublist, window=min(len(i) for i in all_returns), min_count=1) for sublist in notrl_returns]
    notrl_avg_returns = [
        item for sublist in notrl_pre_avg_returns for item in sublist]
    notrl_flat_returns = [
        item for sublist in notrl_returns for item in sublist]
    x_episodes = [i + all_steps[0][-1] - notrl_steps[0][0]
                  for i in notrl_steps[0]]

    # plt.style.use('grayscale')
    fig = plt.figure()
    a0 = fig.add_subplot(1, 1, 1)
    val = 0
    for j, i in enumerate(all_steps):
        if j == len(all_steps) - 1:
            a0.axvline(x=i[-1], linestyle='--',
                       color='#ccc5c6', label='task separator')
        else:
            a0.axvline(x=i[-1], linestyle='--', color='#ccc5c6')
    # a0.plot(flat_steps, flat_returns, label="Incremental Immediate Return")
    # a0.plot(x_episodes, notrl_flat_returns, label="Direct Immediate Return")
    a0.plot(flat_steps, avg_returns, label="incremental averaged return",
            color='#fb7e28', linewidth=1, linestyle='-')  # d94549'
    a0.plot(x_episodes, notrl_avg_returns, label="direct averaged return",
            color='#2678b2', linestyle='-', linewidth=1)  # 2678b2
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    # a0.set_aspect(aspect=500)
    plt.xlabel("Steps")
    plt.ylabel("Return")
    plt.legend(loc="lower right")
    plt.axis([None, None, -12, 1])
    plt.title("Incremental Transfer from Source to Target")
    plt.savefig('9by9.eps', format='eps', dpi=1000)
    plt.show()
