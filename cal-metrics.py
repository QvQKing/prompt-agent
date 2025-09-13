import os
import json
import pandas as pd

# Folder containing the dataset result folders
base_results_dir = 'results-0906-1.5b'

# Initialize a list to store the data
data = []

# Walk through the directories in the base results folder
for dataset_folder in os.listdir(base_results_dir):
    dataset_dir = os.path.join(base_results_dir, dataset_folder)
    
    # Check if it's a directory and has an eval.json file
    if os.path.isdir(dataset_dir):
        eval_file_path = os.path.join(dataset_dir, 'eval.json')
        
        if os.path.exists(eval_file_path):
            # Read the eval.json file
            with open(eval_file_path, 'r', encoding='utf-8') as f:
                eval_data = json.load(f)
                
                # Extract relevant data
                dataset_name = dataset_folder
                count = eval_data.get('count', 0)
                exact_match_mean = eval_data.get('exact_match', {}).get('mean', 0.0)
                exact_match_sum = eval_data.get('exact_match', {}).get('sum', 0)
                substring_match_mean = eval_data.get('substring_match', {}).get('mean', 0.0)
                substring_match_sum = eval_data.get('substring_match', {}).get('sum', 0)
                f1_mean = eval_data.get('f1', {}).get('mean', 0.0)
                semantic_similarity_mean = eval_data.get('semantic_similarity', {}).get('mean', 0.0)
                
                # Append the data to the list
                data.append({
                    'dataset': dataset_name,
                    'count': count,
                    'exact_match_mean': exact_match_mean,
                    'exact_match_sum': exact_match_sum,
                    'substring_match_mean': substring_match_mean,
                    'substring_match_sum': substring_match_sum,
                    'f1_mean': f1_mean,
                    'semantic_similarity_mean': semantic_similarity_mean
                })

# Convert the list of data to a DataFrame
df = pd.DataFrame(data)

# Use the base folder name (e.g., 'results-0906-1.5b') for the CSV file name
output_csv_path = f"{base_results_dir}.csv"
df.to_csv(output_csv_path, index=False, encoding='utf-8')

print(f"Results have been saved to {output_csv_path}")
