import numpy as np
import pandas as pd

environment_rows = 11
environment_columns = 11

# define training parameters
epsilon = 0.4          # the percentage of time when we should take the best action (instead of a random action)
discount_factor = 0.4  # discount factor for future rewards
learning_rate = 0.4    # the rate at which the AI agent should learn
episode = 10000        # no. of times to run

# define starting position
starting_row = 7
starting_column = 9

#define goal position
goal_row = 0
goal_column = 5

# orientation: 0 = 'up', 1 = 'right', 2 = 'down', 3 = 'left'
orientation = ['up', 'right', 'down', 'left']

# q-table initiation
q_values = np.zeros((environment_rows, environment_columns, 4, 3))

# actions defined
actions = ['up', 'right', 'left']

# rewards defined
rewards = np.full((environment_rows, environment_columns), -100.)
rewards[goal_row, goal_column] = 100.

# grid world defined
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

# Helper Functions
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

def get_next_action(current_row_index, current_column_index, orientation, epsilon):
    if np.random.random() < epsilon:
        return np.argmax(q_values[current_row_index, current_column_index, orientation])
    else:
        return np.random.randint(3)

def update_orientation(orientation, action_index):
    if action_index == 2:  # Left turn relative to orientation
        orientation = (orientation - 1) % 4
    elif action_index == 1:  # Right turn relative to orientation
        orientation = (orientation + 1) % 4
    return orientation

def get_next_location(current_row_index, current_column_index, action_index, orientation):
    new_row_index = current_row_index
    new_column_index = current_column_index
    new_orientation = orientation

    if action_index == 0:  # Up
        if orientation == 0 and current_row_index > 0:  # Facing up
            new_row_index -= 1
        elif orientation == 1 and current_column_index < environment_columns - 1:  # Facing right
            new_column_index += 1
        elif orientation == 2 and current_row_index < environment_rows - 1:  # Facing down
            new_row_index += 1
        elif orientation == 3 and current_column_index > 0:  # Facing left
            new_column_index -= 1
    else:  # Turn left or right
        new_orientation = update_orientation(orientation, action_index)

    return new_row_index, new_column_index, new_orientation

def get_shortest_path(start_row_index, start_column_index, start_orientation):
    if is_terminal_state(start_row_index, start_column_index):
        return []
    else:
        current_row_index, current_column_index, orientation = start_row_index, start_column_index, start_orientation
        path = []
        path.append((current_row_index, current_column_index, orientation))

        while not is_terminal_state(current_row_index, current_column_index):
            action_index = np.argmax(q_values[current_row_index, current_column_index, orientation])
            action_taken = actions[action_index]
            new_row_index, new_column_index, new_orientation = get_next_location(current_row_index, current_column_index, action_index, orientation)

            print(f"Visited State: ({current_row_index}, {current_column_index}, Orientation: {orientation}), Action Taken: {action_taken}, New State: ({new_row_index}, {new_column_index}, Orientation: {new_orientation})")

            path.append((new_row_index, new_column_index, new_orientation))
            current_row_index, current_column_index, orientation = new_row_index, new_column_index, new_orientation

        return path



# Training Process
for episode in range(episode):

    row_index, column_index = get_starting_location()
    orientation = 3

    while not is_terminal_state(row_index, column_index):

        action_index = get_next_action(row_index, column_index, orientation, epsilon)
        old_row_index, old_column_index, old_orientation = row_index, column_index, orientation
        row_index, column_index, orientation = get_next_location(row_index, column_index, action_index, orientation)


        reward = rewards[row_index, column_index]
        old_q_value = q_values[old_row_index, old_column_index, old_orientation, action_index]
        temporal_difference = reward + (discount_factor * np.max(q_values[row_index, column_index, orientation])) - old_q_value

        new_q_value = old_q_value + (learning_rate * temporal_difference)
        q_values[old_row_index, old_column_index, old_orientation, action_index] = new_q_value

print('Training complete!')


# Display shortest paths
print('Orientation:', orientation)
print('Start Position:', starting_row, ',', starting_column)
print('Goal Position:', goal_row, ',', goal_column)
print('Path Taken:')
path_taken = get_shortest_path(starting_row, starting_column, orientation)
print("Path:", path_taken)

# create dataset
def store_q_learning_dataset_new(start_row_index, start_column_index, start_orientation, goal_row_index, goal_column_index,
                                 filename='q_learning_dataset_new.csv'):
    data = []

    if is_terminal_state(start_row_index, start_column_index):
        return
    else:
        current_row_index, current_column_index, orientation = start_row_index, start_column_index, start_orientation

        while not is_terminal_state(current_row_index, current_column_index):
            action_index = get_next_action(current_row_index, current_column_index, orientation, 1.0)
            action_taken = actions[action_index]
            new_row_index, new_column_index, new_orientation = get_next_location(current_row_index, current_column_index, action_index, orientation)

            data.append({
                'start state': (start_row_index, start_column_index, start_orientation),
                'state': (current_row_index, current_column_index, orientation),
                'action': action_taken,
                'goal state': (goal_row_index, goal_column_index)
            })

            current_row_index, current_column_index, orientation = new_row_index, new_column_index, new_orientation

    df = pd.DataFrame(data)
    df.to_csv(filename, mode='a', header=not pd.io.common.file_exists(filename), index=False)
    print(f'Dataset appended to {filename}')


store_q_learning_dataset_new(starting_row, starting_column, orientation, goal_row, goal_column)
