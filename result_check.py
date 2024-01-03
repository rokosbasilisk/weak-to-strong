import os
import re

folder_path = "results/"

# Define a regular expression pattern to extract information from the filename
pattern = r"EleutherAI_pythia-(\d+b)_step(\d+)_EleutherAI_pythia-(\db)_step(\d+)_.results_summary.json"

# Iterate through files in the folder
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    # Check if the file is a JSON file and matches the pattern
    if filename.endswith(".json") and re.match(pattern, filename):
        # Extract information using the regular expression groups
        match = re.match(pattern, filename)
        weak_params, weak_steps, strong_params, strong_steps = match.groups()

        # Print or use the extracted information as needed
        print(f"File: {filename}")
        print(f"Weak Params: {weak_params}")
        print(f"Weak Steps: {weak_steps}")
        print(f"Strong Params: {strong_params}")
        print(f"Strong Steps: {strong_steps}")
        print("\n")
