import pandas as pd
import csv

# Load your tab-separated data file
# Make sure to name your columns for easier access
try:
    df = pd.read_csv('data.csv')
    df.columns = [
        'sent_id', 'participant_id', 'premise',
        'hypothesis', 'label', 'explanation', 'participant_type'
    ]
except Exception as e:
    print(f"Error reading file: {e}")
    print("Please ensure your file is named 'data.tsv' and is in the same directory.")
    exit()

# --- Create the Sentence Pairs Table ---
# Select only the unique sentence pairs based on sent_id
sentence_pairs = df[['sent_id', 'premise', 'hypothesis']].drop_duplicates(subset='sent_id').reset_index(drop=True)

# Save to a new CSV
sentence_pairs.to_csv('sentence_pairs.csv', index=False)
print("Successfully created sentence_pairs.csv")
print(sentence_pairs.head())

# --- Create the Predictions Table ---
# Select the prediction-specific columns
predictions = df[['sent_id', 'participant_id', 'participant_type', 'label', 'explanation']]

# Save to a new CSV
predictions.to_csv('predictions.csv', index=False)
print("\nSuccessfully created predictions.csv")
print(predictions.head())