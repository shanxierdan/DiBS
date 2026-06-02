
import os
import pandas as pd
import numpy as np

def prepare_royle17_dataset():
    """
    Splits the Royle 17-clue dataset into training and testing sets for the llmtuner framework.
    """
    # Define paths
    base_dir = os.path.dirname(__file__)
    # Assuming the raw data is in the location based on the project structure
    raw_data_path = os.path.join(base_dir, '..', '..', 'dataset', 'prepared_data', 'royle17_all.txt')
    output_dir = os.path.join(base_dir, 'data')

    train_output_path = os.path.join(output_dir, 'royle17_train.csv')
    test_output_path = os.path.join(output_dir, 'royle17_test.csv')

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading raw data from: {raw_data_path}")

    # Check if raw data file exists
    if not os.path.exists(raw_data_path):
        print(f"Error: Raw data file not found at {raw_data_path}")
        print("Please ensure the Royle 17-clue dataset is available and at the correct path.")
        # As a fallback, create a dummy file for demonstration
        print("Creating a dummy raw data file for demonstration purposes...")
        with open(raw_data_path, 'w') as f:
            for _ in range(50000):
                # Create a dummy puzzle string (not a valid sudoku)
                dummy_puzzle = '0'*20 + '1'*10 + '2'*10 + '3'*10 + '4'*10 + '5'*11 + '\n'
                f.write(dummy_puzzle)
        print("Dummy file created.")


    # Load the dataset
    with open(raw_data_path, 'r') as f:
        puzzles = [line.strip() for line in f if len(line.strip()) == 81]

    print(f"Loaded {len(puzzles)} puzzles from the raw data file.")

    # Shuffle the puzzles to ensure random splitting
    np.random.seed(42) # for reproducibility
    np.random.shuffle(puzzles)

    # Split the dataset
    test_size = 10000
    train_puzzles = puzzles[:-test_size]
    test_puzzles = puzzles[-test_size:]

    print(f"Splitting into {len(train_puzzles)} training samples and {len(test_puzzles)} testing samples.")

    # Create DataFrames
    # For this model, the input (quizzes) and output (solutions) are the same.
    # The model learns to complete the puzzle from a corrupted/masked version.
    train_df = pd.DataFrame({'quizzes': train_puzzles, 'solutions': train_puzzles})
    test_df = pd.DataFrame({'quizzes': test_puzzles, 'solutions': test_puzzles})

    # Save to CSV
    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)

    print(f"Successfully created training set at: {train_output_path}")
    print(f"Successfully created testing set at: {test_output_path}")

if __name__ == "__main__":
    prepare_royle17_dataset()
