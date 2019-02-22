import numpy as np
import matplotlib.pyplot as plt

GAMMA = 0.9

class GridWorld:

    def __init__(self):
        # 10  o
        # o   o
        # S   o

        self.x_lim = 2
        self.y_lim = 3

        # Start position
        self.x_start = 0
        self.y_start = 0

        self.reset_state()

        self.rewards = {
            (0, 2): 1.0,
            (1, 0): -0.1,
            (0, 1): -2.0,
            (1, 1): -0.1,
            (0, 2): -0.1,
            (1, 2): -0.1,
        }

        self.actions = {
            (0, 0): ('U', 'R'),
            (1, 0): ('U'),
            (0, 1): ('U', 'R'),
            (1, 1): ('U', 'L'),
            (1, 2): ('L'),
        }

        self.values = {
            (0, 0): 0,
            (1, 0): 0,
            (0, 1): 0,
            (1, 1): 0,
            (0, 2): 0,
            (1, 2): 0,
        }

        self.policy = {
            (0, 0): 'U',
            (1, 0): 'U',
            (0, 1): 'U',
            (1, 1): 'U',
            (1, 2): 'L',
        }

        self.states = self.all_states()

    def reset_state(self):
        self.x = self.x_start
        self.y = self.y_start

    def set_random_policy(self):
        self.policy = {}
        for state in self.actions.keys():
            # action_index = np.random.random_integers(len(self.actions[state])) - 1
            # self.policy[state] = self.actions[state][action_index]
            self.policy[state] = self.get_random_action(state)

    def get_random_action(self, state):
        action_index = np.random.random_integers(len(self.actions[state])) - 1
        action_to_return = self.actions[state][action_index]
        return action_to_return

    def epsilon_greedy_random_action(self, state, epsilon=0.1):
        p = np.random.random()
        if p < epsilon:
            action = self.get_random_action(state)
        else:
            action = self.policy[state]
        return action

    def reset_values(self):
        self.values = {
            (0, 0): 0,
            (1, 0): 0,
            (0, 1): 0,
            (1, 1): 0,
            (0, 2): 0,
            (1, 2): 0,
        }

    def set_state(self, state):
        self.x = state[0]
        self.y = state[1]

    def current_state(self):
        return (self.x, self.y)

    def all_states(self):
        return set(self.actions.keys()) | set(self.rewards.keys())

    def is_terminal(self, state):
        return state not in self.actions

    def step(self, action):
        if action in self.actions[self.current_state()]:
            if action == 'R':
                self.x += 1
            elif action == 'L':
                self.x -= 1
            elif action == 'U':
                self.y += 1
            elif action == 'D':
                self.y -= 1
        return self.current_state(), self.rewards.get(self.current_state(), 0), self.game_over(self.current_state())
 
    def undo_step(self, action):
        if action == 'R':
            self.x -= 1
        elif action == 'L':
            self.x += 1
        elif action == 'U':
            self.y -= 1
        elif action == 'D':
            self.y += 1
        assert(self.current_state() in self.all_states())

    def game_over(self, state):
        return state not in self.actions

    def print_policy(self):
        for y in range(self.y_lim-1, -1, -1):
            print("------------")
            for x in range(self.x_lim):
                action = self.policy.get((x, y), " ")
                print("  %s  |" % action, end="")
            print("")
        print("------------")

    def print_values(self):
        for  y in range(self.y_lim-1, -1, -1):
            print("------------")
            for x in range(self.x_lim):
                value = self.values.get((x, y), 0)
                if value >= 0:
                    print(" %.2f|" % value, end="")
                else:
                    print("%.2f|" % value, end="")
            print("")
        print("------------")

    def print_rewards(self):
        for  y in range(self.y_lim-1, -1, -1):
            print("------------")
            for x in range(self.x_lim):
                reward = self.rewards.get((x, y), 0)
                if reward >= 0:
                    print(" %.2f|" % reward, end="")
                else:
                    print("%.2f|" % reward, end="")
            print("")
        print("------------")

    def max_dict(self, dict_in):
        max_key = None
        max_value = float('-inf')
        for key, value in dict_in.items():
            if value > max_value:
                max_value = value
                max_key = key
        return max_key, max_value

    def play_game(self):
        num_steps = 0

        # Start state
        self.reset_state()
        done = False
        states_actions_rewards = []

        while (not done) and (num_steps < 5):
            old_state = self.current_state()
            action = self.epsilon_greedy_random_action(self.current_state())
            state, reward, done = self.step(action)
            num_steps = num_steps + 1
            states_actions_rewards.append((old_state, action, reward))
            # print(old_state, action, reward)
        # print("Episode end")
        # print(states_actions_rewards)

        G = 0
        states_actions_returns = []

        for state, action, reward in reversed(states_actions_rewards):
            print(state, action, reward)
            G = reward + GAMMA * G
            states_actions_returns.append((state, action, G))
            
        states_actions_returns.reverse()

        return states_actions_returns


if __name__ == "__main__":

    num_ep = 25

    grid = GridWorld()

    print("Rewards:")
    grid.print_rewards()

    print("Values:")
    grid.print_values()

    print("Policy:")
    grid.set_random_policy()
    grid.print_policy()

    Q = {}
    returns = {}

    for state in grid.states:
        if state in grid.actions:
            Q[state] = {}
            for action in grid.actions[state]:
                Q[state][action] = 0
                returns[(state, action)] = []
        else:
            pass


    deltas = []
    for t in range(num_ep):
        biggest_change = 0
        states_actions_returns = grid.play_game()

        seen_state_action_pairs = set()
        for state, action, G in states_actions_returns:
            if not grid.is_terminal(state):
                sa = (state, action)
                if sa not in seen_state_action_pairs:
                    old_q = Q[state][action]
                    returns[sa].append(G)
                    Q[state][action] = np.mean(returns[sa])
                    biggest_change = max(biggest_change, np.abs(old_q - Q[state][action]))
                    seen_state_action_pairs.add(sa)
        deltas.append(biggest_change)

        for state in grid.policy.keys():
            action, _ = grid.max_dict(Q[state])
            grid.policy[state] = action

    # plt.plot(deltas)
    # plt.show()

    grid.values = {}
    for state in grid.policy.keys():
        # print(state)
        action, value = grid.max_dict(Q[state])
        grid.values[state] = value

    print("Final values:")
    # print(grid.values)
    grid.print_values()
    # print(Q)
    print("Final policy:")
    grid.print_policy()
    # print(grid.policy)