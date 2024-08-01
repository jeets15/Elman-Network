import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

environment_rows = 11
environment_columns = 11

q_values = np.zeros((environment_rows, environment_columns, 4))

actions = ['up', 'right', 'down', 'left']

rewards = np.full((environment_rows, environment_columns), -100.)
rewards[0, 5] = 100.  # set the reward for the packaging area (i.e., the goal) to 100

aisles = {}  # store locations in a dictionary
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
            current_row_index, current_column_index = get_next_location(current_row_index, current_column_index, action_index)
            shortest_path.append([current_row_index, current_column_index])
        return shortest_path

# define training parameters
epsilon = 0.6  # the percentage of time when we should take the best action (instead of a random action)
discount_factor = 0.6  # discount factor for future rewards
learning_rate = 0.6  # the rate at which the AI agent should learn

for episode in range(1000):
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

print(get_shortest_path(3, 9))
print(get_shortest_path(5, 0))
print(get_shortest_path(9, 5))

def visualize_q_values(q_values):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    action_titles = ['Up', 'Right', 'Down', 'Left']
    for i in range(4):
        ax = axes[i]
        cax = ax.matshow(q_values[:, :, i], cmap='viridis')
        fig.colorbar(cax, ax=ax)
        ax.set_title(f'Q-values for action: {action_titles[i]}')
        ax.set_xlabel('Columns')
        ax.set_ylabel('Rows')
    plt.tight_layout()
    plt.show()

def extract_policy(q_values):
    policy = np.full((environment_rows, environment_columns), ' ')
    for row in range(environment_rows):
        for col in range(environment_columns):
            if not is_terminal_state(row, col):
                best_action = np.argmax(q_values[row, col])
                policy[row, col] = actions[best_action][0].upper()
    return policy

# Visualize the Q-values
visualize_q_values(q_values)

# Extract and print the policy
policy = extract_policy(q_values)
print("Learned Policy:")
print(policy)




def get_best_action_from_q_values(row, col):
    best_action_index = np.argmax(q_values[row, col])
    return actions[best_action_index][0].upper()



# New code starts here to create the dataset in the specified format
def create_q_value_dataset():
    dataset = []

    for row in range(environment_rows):
        for col in range(environment_columns):
            for action_index in range(len(actions)):
                state = np.array([row, col])
                action = action_index
                q_value = q_values[row, col, action_index]
                dataset.append((state, action, q_value))

    return dataset

# Create the dataset
q_value_dataset = create_q_value_dataset()

# Display the first few entries of the dataset
print(q_value_dataset)

# If you want to convert it to a DataFrame for further analysis or visualization
df = pd.DataFrame(q_value_dataset, columns=['State', 'Action', 'Q-Value'])

# Visualize the DataFrame using matplotlib
def visualize_q_value_dataset(df):
    fig, ax = plt.subplots(figsize=(14, 8))
    for action in range(len(actions)):
        action_data = df[df['Action'] == action]
        sc = ax.scatter(
            action_data['State'].apply(lambda x: x[1]),  # Column index
            action_data['State'].apply(lambda x: x[0]),  # Row index
            c=action_data['Q-Value'],
            cmap='viridis',
            label=actions[action],
            s=100,
            alpha=0.6,
            edgecolors='w'
        )
    plt.colorbar(sc, ax=ax, label='Q-Value')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_title('Q-Values for Actions in Each State')
    ax.legend()
    plt.gca().invert_yaxis()
    plt.show()

# Visualize the Q-value dataset
visualize_q_value_dataset(df)

# Convert the dataset to a DataFrame for further analysis or visualization
df = pd.DataFrame(q_value_dataset, columns=['State', 'Action', 'Q-Value'])

# Display the DataFrame as a vertical table
print(df)

# Save the dataset to a CSV file
csv_file_path = "q_value_dataset.csv"
df.to_csv(csv_file_path, index=False)
print(f"Dataset saved as CSV to {csv_file_path}")

