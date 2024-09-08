import pandas as pd

# action to vectors
action_mapping = {
    "right": (1, 0, 0, 0),
    "forward": (0, 1, 0, 0),
    "left": (0, 0, 1, 0),
    "backward": (0, 0, 0, 1)
}

# states to vectors
single_state_mapping = {
    0: (0, 0, 0, 0),
    1: (0, 0, 0, 1),
    2: (0, 0, 1, 0),
    3: (0, 0, 1, 1),
    4: (0, 1, 0, 0),
    5: (0, 1, 0, 1),
    6: (0, 1, 1, 0),
    7: (0, 1, 1, 1),
    8: (1, 0, 0, 0),
    9: (1, 0, 0, 1)
}


# encoding state
def encode_state(state):
    x, y = map(int, state.strip('()').split(', '))
    state_vector_x = single_state_mapping.get(x, (0, 0, 0, 0))
    state_vector_y = single_state_mapping.get(y, (0, 0, 0, 0))
    concatenated_state = state_vector_x + state_vector_y
    return concatenated_state

# processing dataset
def process_q_learning_dataset(csv_file):
    df = pd.read_csv(csv_file)
    samples = []

    for _, row in df.iterrows():
        state = row['state']
        action = row['action']
        encoded_state = encode_state(state)
        action_vector = action_mapping.get(action.lower(), (0, 0, 0, 0))
        samples.append((encoded_state, action_vector))

    return samples


# storing vectorised dataset
def store_preprocessed_q_dataset(samples, filename='preprocessed_q_dataset.csv'):
    with open(filename, 'a') as f:  # Open in append mode
        for i, (encoded_state, action_vector) in enumerate(samples):
            # Format the line exactly as required
            line = f"samples[{i}] = {encoded_state}, {action_vector}\n"
            f.write(line)
    print(f"Dataset appended to {filename}")


if __name__ == "__main__":
    csv_file = 'q_learning_dataset.csv'
    samples = process_q_learning_dataset(csv_file)
    for i, sample in enumerate(samples):
        print(f'samples[{i}] = {sample[0]}, {sample[1]}')

    store_preprocessed_q_dataset(samples)