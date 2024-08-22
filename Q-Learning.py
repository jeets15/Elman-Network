import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# define gridworld environment
rows = 11
columns = 11

# define training parameters
epsilon = 0.4          # the percentage of time when we should take the best action instead of a random action
discount_factor = 0.4  # discount factor for future rewards
learning_rate = 0.4    # the rate at which the agent should learn
episode = 3000

# define starting position
starting_row = 9
starting_column = 0

#define goal position
goal_row = 0
goal_column = 5

# Initialise q- value table
# as rows, columns and no. of action as input
q_values = np.zeros((rows, columns, 4))

# define the actions
actions = ['up', 'right', 'down', 'left']

# set the reward as 100 for goal state and -100 for terminal states
rewards = np.full((rows, columns), -100.)
rewards[goal_row, goal_column] = 100.

# define the non-terminal states
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

# function to check if the state is terminal
def terminal_state(current_row_index, current_column_index):

  if rewards[current_row_index, current_column_index] == -1.:
    return False
  else:
    return True

# function to get a starting location
def starting_location():

  current_row_index = np.random.randint(rows)
  current_column_index = np.random.randint(columns)

  while terminal_state(current_row_index, current_column_index):
    current_row_index = np.random.randint(rows)
    current_column_index = np.random.randint(columns)
  return current_row_index, current_column_index


# function to get the next action
def next_action(current_row_index, current_column_index, epsilon):

  if np.random.random() < epsilon:
    return np.argmax(q_values[current_row_index, current_column_index])
  else:
    return np.random.randint(4)


# function to get the next action
def next_location(current_row_index, current_column_index, action_index):
  new_row_index = current_row_index
  new_column_index = current_column_index
  if actions[action_index] == 'up' and current_row_index > 0:
    new_row_index -= 1
  elif actions[action_index] == 'right' and current_column_index < columns - 1:
    new_column_index += 1
  elif actions[action_index] == 'down' and current_row_index < rows - 1:
    new_row_index += 1
  elif actions[action_index] == 'left' and current_column_index > 0:
    new_column_index -= 1
  return new_row_index, new_column_index


# function to compute the shortest path
def shortest_path(start_row_index, start_column_index):
    if terminal_state(start_row_index, start_column_index):
        return []
    else:
        current_row_index, current_column_index = start_row_index, start_column_index
        shortest_path = []
        shortest_path.append([current_row_index, current_column_index])

        while not terminal_state(current_row_index, current_column_index):
            action_index = next_action(current_row_index, current_column_index, 1.)

            # Get the name of the action taken
            action_taken = actions[action_index]

            # Move to the next location based on the action
            new_row_index, new_column_index = next_location(current_row_index, current_column_index, action_index)

            # Print the state visited and the action taken
            print(f"Visited State: ({current_row_index}, {current_column_index}), Action Taken: {action_taken}")

            # Update the current position
            current_row_index, current_column_index = new_row_index, new_column_index

            # Add the new position to the path
            shortest_path.append([current_row_index, current_column_index])

        return shortest_path



# Training Process
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


# Display shortest paths
print('Start Position:', starting_row, ',', starting_column)
print('Goal Position:', goal_row, ',', goal_column)
print('Path Taken:')
path_taken = shortest_path(starting_row, starting_column)
print("Path:", path_taken)


# Storing CSV file
def dataset(start_row_index, start_column_index, goal_row_index, goal_column_index, filename='q_learning_dataset.csv'):
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
                'start state': (start_row_index, start_column_index),
                'goal state': (goal_row_index, goal_column_index),
                'state visited': (current_row_index, current_column_index),
                'action taken': action_taken
            })


            current_row_index, current_column_index = new_row_index, new_column_index


    df = pd.DataFrame(data)


    if not os.path.isfile(filename):
        df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, mode='a', header=False, index=False)

    print(f'Dataset appended to {filename}')


dataset(starting_row, starting_column, goal_row, goal_column)


#Visualise Grid World
def visualize(rewards, goal_position, start_position):
    environment_rows, environment_columns = rewards.shape
    grid = np.zeros_like(rewards)

    grid[rewards == -100.] = 0  # Obstacles
    grid[rewards == -1.] = 1    # Walkable paths
    grid[goal_position] = 2     # Goal
    grid[start_position] = 3    # Start

    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap='gray', origin='upper')

    plt.imshow(np.where(grid == 2, 1, np.nan), cmap='YlOrBr', vmin=0, vmax=1, origin='upper')

    plt.imshow(np.where(grid == 3, 1, np.nan), cmap='PiYG', vmin=0, vmax=1, origin='upper')

    plt.grid(which='both', color='black', linestyle='-')

    plt.xticks(np.arange(environment_columns), labels=np.arange(environment_columns))
    plt.yticks(np.arange(environment_rows), labels=np.arange(environment_rows))

    plt.show()


goal_position = (goal_row, goal_column)
start_position = (starting_row, starting_column)  # Update with your actual starting position


visualize(rewards, goal_position, start_position)
