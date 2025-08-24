import pandas as pd
import glob
import os

# List of your 3 folders
folders = [
    r"Data\Novels",
    r"Data\Songs",
    r"Data\Subtitles"
]

all_files = []

# Loop over each folder
for folder in folders:
    csv_files = glob.glob(os.path.join(folder, "*.xlsx"))
    for file in csv_files:
        df = pd.read_excel(file)
        all_files.append(df)

# Concatenate all data
final_df = pd.concat(all_files, ignore_index=True)

# Save to one big CSV
final_df.to_csv("all_data.csv", index=False)