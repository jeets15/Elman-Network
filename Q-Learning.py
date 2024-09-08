import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# define gridworld environment
environment_rows = 11
environment_columns = 11

# define training parameters
epsilon = 0.4          # the percentage of time when we should take the best action (instead of a random action)
discount_factor = 0.4  # discount factor for future rewards
learning_rate = 0.4    # the rate at which the AI agent should learn
episode = 3000

# define starting position
starting_row = 9
starting_column = 0

#define goal position
goal_row = 0
goal_column = 5

# initialise q- value table
q_values = np.zeros((environment_rows, environment_columns, 4))

# define actions
actions = ['forward', 'right', 'backward', 'left']

# define rewards
rewards = np.full((environment_rows, environment_columns), -100.)
rewards[goal_row, goal_column] = 100.

# define non-terminal states
rewards = np.full((environment_rows, environment_columns), -100.)
rewards[goal_row, goal_column] = 100.


aisles = {}
aisles[1] = [i for i in range(1, 10)]
aisles[2] = [1, 7, 9]
aisles[3] = [i for i in range(1, 8)]
aisles[3].append(9)
aisles[4] = [3, 7]
aisles[5] = [i for i in range(11)]
aisles[6] = [5]
aisles[7] = [i for i in range(1, 10)]
aisles[8] = [3, 7]
aisles[9] = [i for i in range(11)]

for row_index in range(1, 10):
    for column_index in aisles[row_index]:
        rewards[row_index, column_index] = -1.


for row in rewards:
    print(row)


# helper functions
def terminal_state(current_row_index, current_column_index):

  if rewards[current_row_index, current_column_index] == -1.:
    return False
  else:
    return True


def starting_location():

  current_row_index = np.random.randint(environment_rows)
  current_column_index = np.random.randint(environment_columns)

  while terminal_state(current_row_index, current_column_index):
    current_row_index = np.random.randint(environment_rows)
    current_column_index = np.random.randint(environment_columns)
  return current_row_index, current_column_index


def next_action(current_row_index, current_column_index, epsilon):

  if np.random.random() < epsilon:
    return np.argmax(q_values[current_row_index, current_column_index])
  else:
    return np.random.randint(4)


def next_location(current_row_index, current_column_index, action_index):
  new_row_index = current_row_index
  new_column_index = current_column_index
  if actions[action_index] == 'forward' and current_row_index > 0:
    new_row_index -= 1
  elif actions[action_index] == 'right' and current_column_index < environment_columns - 1:
    new_column_index += 1
  elif actions[action_index] == 'backward' and current_row_index < environment_rows - 1:
    new_row_index += 1
  elif actions[action_index] == 'left' and current_column_index > 0:
    new_column_index -= 1
  return new_row_index, new_column_index


def shortest_path(start_row_index, start_column_index):
    if terminal_state(start_row_index, start_column_index):
        return []
    else:
        current_row_index, current_column_index = start_row_index, start_column_index
        shortest_path = []
        shortest_path.append([current_row_index, current_column_index])

        while not terminal_state(current_row_index, current_column_index):
            action_index = next_action(current_row_index, current_column_index, 1.)
            action_taken = actions[action_index]
            new_row_index, new_column_index = next_location(current_row_index, current_column_index, action_index)
            print(f"Visited State: ({current_row_index}, {current_column_index}), Action Taken: {action_taken}")
            current_row_index, current_column_index = new_row_index, new_column_index
            shortest_path.append([current_row_index, current_column_index])

        return shortest_path


# Training
for episode in range(episode):
    row_index, column_index = starting_location()
    while not terminal_state(row_index, column_index):
        action_index = next_action(row_index, column_index, epsilon)
        old_row_index, old_column_index = row_index, column_index
        row_index, column_index = next_location(row_index, column_index, action_index)
        reward = rewards[row_index, column_index]
        old_q_value = q_values[old_row_index, old_column_index, action_index]
        temporal_difference = reward + (discount_factor * np.max(q_values[row_index, column_index])) - old_q_value
        new_q_value = old_q_value + (learning_rate * temporal_difference)
        q_values[old_row_index, old_column_index, action_index] = new_q_value

print('Training complete!')



# display paths
print('Start Position:', starting_row, ',', starting_column)
print('Goal Position:', goal_row, ',', goal_column)
print('Path Taken:')
path_taken = shortest_path(starting_row, starting_column)
print("Path:", path_taken)


# storing CSV file
def store_q_learning_dataset(start_row_index, start_column_index, goal_row_index, goal_column_index, filename='q_learning_dataset.csv'):
    data = []

    if terminal_state(start_row_index, start_column_index):
        return
    else:
        current_row_index, current_column_index = start_row_index, start_column_index

        while not terminal_state(current_row_index, current_column_index):
            action_index = next_action(current_row_index, current_column_index, epsilon=1.0)
            action_taken = actions[action_index]
            new_row_index, new_column_index = next_location(current_row_index, current_column_index, action_index)


            data.append({
                'state': (current_row_index, current_column_index),
                'action': action_taken
            })

            current_row_index, current_column_index = new_row_index, new_column_index
    df = pd.DataFrame(data)


    if not os.path.isfile(filename):
        df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, mode='a', header=False, index=False)

    print(f'Dataset appended to {filename}')


store_q_learning_dataset(starting_row, starting_column, goal_row, goal_column)


#visualise grid world
def visualize_environment(rewards, goal_position, start_position):
    environment_rows, environment_columns = rewards.shape
    grid = np.zeros_like(rewards)

    # Assign colors for different cell types
    grid[rewards == -100.] = 0  # Obstacles
    grid[rewards == -1.] = 1    # paths
    grid[goal_position] = 2     # Goal
    grid[start_position] = 3    # Start

    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap='gray', origin='upper')
    plt.imshow(np.where(grid == 2, 1, np.nan), cmap='YlOrBr', vmin=0, vmax=1, origin='upper')
    plt.imshow(np.where(grid == 3, 1, np.nan), cmap='PiYG', vmin=0, vmax=1, origin='upper')

    # Set ticks and add black grid lines correctly
    plt.xticks(np.arange(environment_columns), labels=np.arange(environment_columns))
    plt.yticks(np.arange(environment_rows), labels=np.arange(environment_rows))
    plt.gca().set_xticks(np.arange(0.5, environment_columns, 1), minor=True)
    plt.gca().set_yticks(np.arange(0.5, environment_rows, 1), minor=True)
    plt.grid(which='minor', color='black', linestyle='-', linewidth=1)

    plt.show()

goal_position = (goal_row, goal_column)
start_position = (starting_row, starting_column)

visualize_environment(rewards, goal_position, start_position)
