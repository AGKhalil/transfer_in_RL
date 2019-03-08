"""Script for running experiment.
"""
from grid_world import GridWorld
from qlearn import QLearn
import sys
import numpy as np
import random
import itertools
import matplotlib.pyplot as plt
import os


def do_task(qlearn, grid, task, exploit=False):

    # number of maximum episodes to run
    nEp = 200

    # initialize algorithm parameters
    old_mean = 0
    delta = 0.000001
    steps = 0
    plt_steps = 0
    returns = 0
    list_returns = []
    list_steps = []
    list_episodes = []
    qlearn_saves = []

    # print("Currently at task ", task)
    for episode in range(1, nEp):
        episode_return = 0
        state = grid.reset_state()
        for step in itertools.count():
            action = qlearn.epsilon_greedy_random_action(state, step, exploit)
            new_state, reward = qlearn.take_step(state, action)
            returns += reward
            episode_return += reward
            new_action = qlearn.epsilon_greedy_random_action(
                new_state, step, exploit)
            qlearn.update_Q(state, action, new_state, reward)
            qlearn_saves.append([state, action, reward, new_state])

            # if step % 100 == 0:
            # print("Task", task, "Episode", episode, "Step", step, "Return", episode_return, "State", state, "Action", grid.arrow(action))            # qlearn.print_values()
            # grid.show_policy(qlearn.policy)
            # qlearn.print_values()

            if qlearn.c_map[new_state]['done'] or step == 200:
                steps += step
                list_returns.append(episode_return)
                list_steps.append(steps)
                list_episodes.append(episode)
                if grid.world == 4:
                    csv_str = "tmp_data/w4x4/s_%s_%d.csv" % (task, episode)
                elif grid.world == 9:
                    csv_str = "tmp_data/w9x9/s_%s_%d.csv" % (task, episode)
                np.savetxt(csv_str, qlearn_saves, fmt="%s", delimiter=",")
                break
            else:
                state = new_state

        current_mean = abs(np.mean(list(np.sum(qlearn.Q.values()))))
        if np.abs(old_mean - current_mean) < delta or episode == nEp - 1:
            # print("Convergence at episode ", episode)
            # print("Total of steps ", steps)
            # print("Cumulative return", returns)
            return qlearn.Q, list_returns, list_episodes, list_steps
        else:
            old_mean = current_mean


def main(iteration):
    world = 4

    # saving directories
    window = 5 # moving mean window
    main_dir = 'qlearn_plots'
    sub_dir = ['4by4can', '4by4nocan', '9by9']
    sub_sub_dir = ['steps', 'episodes']

    for sub_d in sub_dir:
        for ss_d in sub_sub_dir:
            dir_name = '/'.join([main_dir, sub_d, 'win' + str(window), ss_d])
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

    # print("-" * 100)

    # Evaluation
    tot_steps = 0
    all_returns = []
    all_steps = []
    all_episodes = []

    notrl_tot_steps = 0
    notrl_returns = []
    notrl_steps = []
    notrl_episodes = []

    # create grid-world instance
    if world == 4:
        canyon = False
        grid = GridWorld(world, canyon)
        if canyon:
            canyon_str = "(CANYON)"
        else:
            canyon_str = "(NO CANYON)"
    elif world == 9:
        canyon_str = ''
        grid = GridWorld(9)
    grid.make_maps()

    possible_actions = grid.possible_actions
    grid.list_of_maps.reverse()

    # Direct learning on final grid
    # print("Direct learning on final grid")
    qlearn = QLearn(grid.final_grid, possible_actions, world)
    Q, returns, episodes, steps = do_task(
        qlearn, grid, len(grid.list_of_maps) - 1)
    notrl_returns.append(returns)
    notrl_steps.append(steps)
    notrl_episodes.append(episodes)
    notrl_tot_steps += steps[-1]
    # print("-" * 80)

    # Incremental transfer learning
    # print("Incremental transfer learning", canyon_str)
    Q = None
    for task, current_map in enumerate(grid.list_of_maps, 0):
        # print("-" * 50)
        # creates qlearn instance
        exploit = False if task == 0 else False
        qlearn = QLearn(current_map, possible_actions, world, Q)
        Q, returns, episodes, steps = do_task(qlearn, grid, task, exploit)
        all_returns.append(returns)
        tot_counter = 0
        epi_counter = 0
        if task != 0:
            tot_counter += all_steps[task - 1][-1]
            epi_counter += all_episodes[task - 1][-1]
            all_steps.append([i + tot_counter for i in steps])
            all_episodes.append([i + epi_counter for i in episodes])
        else:
            all_steps.append([i for i in steps])
            all_episodes.append([i for i in episodes])
    # print("-" * 100)

    # print("Incremental Transfer Cumulative total of steps",
          # all_steps[-1][-1] - all_steps[0][-1])
    # print("Direct Cumulative total of steps", notrl_steps[-1][-1])

    flat_episodes = [item for sublist in all_episodes for item in sublist]
    flat_returns = [item for sublist in all_returns for item in sublist]
    flat_steps = [item for sublist in all_steps for item in sublist]
    tmp_array = np.array(flat_returns)
    notrl_avg_returns = []
    avg_returns = []
    for t in range(len(flat_returns)):
        avg_returns.append(tmp_array[max(0, t - window):(t + 1)].mean())
    notrl_flat_returns = [
        item for sublist in notrl_returns for item in sublist]
    tmp_array_1 = np.array(notrl_flat_returns)
    for t in range(len(notrl_flat_returns)):
        notrl_avg_returns.append(
            tmp_array_1[max(0, t - window):(t + 1)].mean())

    fig = plt.figure()
    a0 = fig.add_subplot(1, 1, 1)
    val = 0
    for j, i in enumerate(all_steps):
        if j == len(all_steps) - 1:
            a0.axvline(x=i[-1], linestyle='--',
                       color='#ccc5c6', label='Task Switch')
        else:
            a0.axvline(x=i[-1], linestyle='--', color='#ccc5c6')
    a0.plot(flat_steps, avg_returns, label="Task Interpolation",
            color='#d73236', linewidth=1, linestyle='-')
    x_steps = [i + all_steps[0][-1] - notrl_steps[0][0]
               for i in notrl_steps[0]]
    a0.plot(x_steps, notrl_avg_returns, label="Tabula Rasa",
            color='#80bbe5', linestyle='-', linewidth=1)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.xlabel("Steps")
    plt.ylabel("Accumulated Reward")
    plt.legend(loc="lower right")
    plt.axis([None, None, -20, 1])
    if world == 4:
        if canyon:
            step_save = 'qlearn_plots/4by4can/' + 'win' + str(window) + '/steps/4by4_canyon_steps'
            plt_title = '4x4 Maze Canyon'
        else:
            step_save = 'qlearn_plots/4by4nocan/' + 'win' + str(window) + '/steps/4by4_nocanyon_steps'
            plt_title = '4x4 Maze Non-Canyon'
    elif world == 9:
        step_save = 'qlearn_plots/9by9/' + 'win' + str(window) + '/steps/9by9_steps'
        plt_title = '9x9 Maze'
    plt.title(plt_title)
    plt.savefig(step_save + iteration + '.eps', format='eps', dpi=1000)
    # fig.show()

    fig1 = plt.figure()
    a1 = fig1.add_subplot(1, 1, 1)
    val = 0
    for j, i in enumerate(all_episodes):
        if j == len(all_episodes) - 1:
            a1.axvline(x=i[-1], linestyle='--',
                       color='#ccc5c6', label='Task Switch')
        else:
            a1.axvline(x=i[-1], linestyle='--', color='#ccc5c6')
    a1.plot(flat_episodes, avg_returns, label="Task Interpolation",
            color='#d73236', linewidth=1, linestyle='-')
    x_episodes = [i + all_episodes[0][-1] - notrl_episodes[0][0]
                  for i in notrl_episodes[0]]
    a1.plot(x_episodes, notrl_avg_returns, label="Tabula Rasa",
            color='#80bbe5', linestyle='-', linewidth=1)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.xlabel("Episodes")
    plt.ylabel("Accumulated Reward")
    plt.legend(loc="lower right")
    plt.axis([None, None, -20, 1])
    plt.title(plt_title)
    if world == 4:
        if canyon:
            epi_save = 'qlearn_plots/4by4can/' + 'win' + str(window) + '/episodes/4by4_canyon_episodes'
        else:
            epi_save = 'qlearn_plots/4by4nocan/' + 'win' + str(window) + '/episodes/4by4_nocanyon_episodes'
    elif world == 9:
        epi_save = 'qlearn_plots/9by9/' + 'win' + str(window) + '/episodes/9by9_episodes'
    plt.savefig(epi_save + iteration + '.eps', format='eps', dpi=1000)
    # fig1.show()

    # input()

if __name__ == "__main__":
    for iteration in range(10):
        print('Iteration', iteration)
        main(str(iteration))
