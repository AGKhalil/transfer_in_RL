import sys
import numpy as np
import itertools
import copy
import random
import matplotlib.pyplot as plt
from collections import defaultdict
from my_grid import GridWorld

class SARSA:

	def __init__(self, c_map, possible_actions, x_lim, y_lim):
		self.alpha = 0.5
		self.discount_factor = 0.9
		self.epsilon = 0.1

		self.c_map = c_map
		self.possible_actions = possible_actions
		self.x_lim = x_lim
		self.y_lim = y_lim

		self.Q = dict()
		self.policy = dict()

	def epsilon_greedy_random_action(self, state):
		p = np.random.random()
		count = 0
		if p < self.epsilon:
			action = random.choice(self.possible_actions)
		else:
			q_all = [self.Q.get((state, a), 0.0) for a in self.possible_actions]
			max_a = [a for a in self.possible_actions if q_all[a] == max(q_all)]
			if len(max_a) > 1:
				action = random.choice(max_a)
			else:
				action = max_a[0]
		self.policy[state] = action
		return action

	def update_Q(self, state, action, new_state, new_action, reward):
		if self.Q.get((state, action), None):
			q = self.Q[(state), action]
			self.Q[state, action] = q + self.alpha * (reward + self.discount_factor * self.Q.get((new_state, action), 0.0) - q)
		else:
			self.Q[state, action] = reward

	def take_step(self, state, action):
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
		for  y in range(self.y_lim-1, -1, -1):
			print('-'*self.x_lim*9)
			for x in range(self.x_lim):
				value = max([self.Q.get(((x, y), a), 0.0) for a in self.possible_actions])
				if value >= 0:
					print(' {:.4f}'.format(value), "|", end="")
				else:
					print('{:.4f}'.format(value), "|", end="")
			print("")
		print('-'*self.x_lim*9)

if __name__ == "__main__":

	nEp = 200
	grid = GridWorld()
	grid.make_maps()
	current_map = grid.list_of_maps[0] # Change index to get different maps 0-4
	
	if not current_map:
		print("Map index is out of range.")
		sys.exit()

	possible_actions = grid.possible_actions
	x_lim, y_lim = grid.x_lim, grid.y_lim

	sarsa = SARSA(current_map, possible_actions, x_lim, y_lim)
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

	print("Environment Map")
	grid.show_grid(sarsa.c_map)
	print("Environment Values")
	sarsa.print_values()
	print("Environment Policy")
	grid.show_policy(sarsa.policy)

