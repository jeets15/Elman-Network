
import numpy as np
import pandas as pd
import os


environment_rows = 11
environment_columns = 11

# define training parameters
epsilon = 0.4  # the percentage of time when we should take the best action (instead of a random action)
discount_factor = 0.4  # discount factor for future rewards
learning_rate = 0.4 # the rate at which the AI agent should learn
episode = 3000

# define starting position
starting_row = 5
starting_column = 10

#define goal position
goal_row = 0
goal_column = 5

q_values = np.zeros((environment_rows, environment_columns, 4))


actions = ['up', 'right', 'down', 'left']


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


def is_terminal_state(current_row_index, current_column_index):

  if rewards[current_row_index, current_column_index] == -1.:
    return False
  else:
    return True


def get_starting_location():

  current_row_index = np.random.randint(environment_rows)
  current_column_index = np.random.randint(environment_columns)

  while is_terminal_state(current_row_index, current_column_index):
    current_row_index = np.random.randint(environment_rows)
    current_column_index = np.random.randint(environment_columns)
  return current_row_index, current_column_index


def get_next_action(current_row_index, current_column_index, epsilon):

  if np.random.random() < epsilon:
    return np.argmax(q_values[current_row_index, current_column_index])
  else:
    return np.random.randint(4)


def get_next_location(current_row_index, current_column_index, action_index):
  new_row_index = current_row_index
  new_column_index = current_column_index
  if actions[action_index] == 'up' and current_row_index > 0:
    new_row_index -= 1
  elif actions[action_index] == 'right' and current_column_index < environment_columns - 1:
    new_column_index += 1
  elif actions[action_index] == 'down' and current_row_index < environment_rows - 1:
    new_row_index += 1
  elif actions[action_index] == 'left' and current_column_index > 0:
    new_column_index -= 1
  return new_row_index, new_column_index


def get_shortest_path(start_row_index, start_column_index):
    if is_terminal_state(start_row_index, start_column_index):
        return []
    else:
        current_row_index, current_column_index = start_row_index, start_column_index
        shortest_path = []
        shortest_path.append([current_row_index, current_column_index])

        while not is_terminal_state(current_row_index, current_column_index):
            action_index = get_next_action(current_row_index, current_column_index, 1.)

            # Get the name of the action taken
            action_taken = actions[action_index]

            # Move to the next location based on the action
            new_row_index, new_column_index = get_next_location(current_row_index, current_column_index, action_index)

            # Print the state visited and the action taken
            print(f"Visited State: ({current_row_index}, {current_column_index}), Action Taken: {action_taken}")

            # Update the current position
            current_row_index, current_column_index = new_row_index, new_column_index

            # Add the new position to the path
            shortest_path.append([current_row_index, current_column_index])

        return shortest_path


# Training Process
for episode in range(episode):

    row_index, column_index = get_starting_location()


    while not is_terminal_state(row_index, column_index):

        action_index = get_next_action(row_index, column_index, epsilon)


        old_row_index, old_column_index = row_index, column_index
        row_index, column_index = get_next_location(row_index, column_index, action_index)


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
path_taken = get_shortest_path(starting_row, starting_column)
print("Path:", path_taken)


def store_q_learning_dataset(start_row_index, start_column_index, goal_row_index, goal_column_index,
                             filename='q_learning_dataset.csv'):
    data = []

    if is_terminal_state(start_row_index, start_column_index):
        return
    else:
        current_row_index, current_column_index = start_row_index, start_column_index

        while not is_terminal_state(current_row_index, current_column_index):
            action_index = get_next_action(current_row_index, current_column_index, 1.)
            action_taken = actions[action_index]
            new_row_index, new_column_index = get_next_location(current_row_index, current_column_index, action_index)

            # Record the data
            data.append({
                'start state': (start_row_index, start_column_index),
                'state': (current_row_index, current_column_index),
                'action': action_taken,
                'goal state': (goal_row_index, goal_column_index)
            })

            current_row_index, current_column_index = new_row_index, new_column_index

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Append to the CSV file without overwriting
    df.to_csv(filename, mode='a', header=not pd.io.common.file_exists(filename), index=False)
    print(f'Dataset appended to {filename}')

store_q_learning_dataset(starting_row, starting_column, goal_row, goal_column)