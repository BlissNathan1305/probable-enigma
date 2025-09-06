import pandas as pd
import numpy as np

# Load the entire sheet
file_path = "Eddie.xlsx"
sheet_name = "parameters"

df = pd.read_excel("Eddie.xlsx", sheet_name=sheet_name, header=None)  # Read without assuming header

# Helper: Find dataset boundaries (assuming datasets are separated by empty rows)
def find_dataset_ranges(df):
    ranges = []
    start = None
    for i in range(len(df)):
        # Check if row is not empty (at least one non-null value)
        if df.iloc[i].notna().any():
            if start is None:
                start = i
        else:
            if start is not None:
                ranges.append((start, i))
                start = None
    # Catch last dataset if sheet doesn't end with empty row
    if start is not None:
        ranges.append((start, len(df)))
    return ranges

ranges = find_dataset_ranges(df)

# Assume first row of each dataset is the header (adjust if needed)
datasets = []
for idx, (start, end) in enumerate(ranges):
    dataset_df = df.iloc[start:end].reset_index(drop=True)
    
    # Use first row as header
    dataset_df.columns = dataset_df.iloc[0]
    dataset_df = dataset_df[1:].reset_index(drop=True)
    
    # For the second dataset (index 1), reshape it
    if idx == 1:
        # 💡 ADJUST THIS PART BASED ON ACTUAL STRUCTURE
        # Example: if it's transposed
        if dataset_df.shape[0] < dataset_df.shape[1]:
            dataset_df = dataset_df.T
            dataset_df.columns = dataset_df.iloc[0]
            dataset_df = dataset_df[1:].reset_index(drop=True)
        
        # Or if columns are shifted, realign them
        expected_columns = datasets[0].columns.tolist()  # From first dataset
        if list(dataset_df.columns) != expected_columns:
            # Try to match by column names or reassign
            dataset_df.columns = expected_columns[:len(dataset_df.columns)]
            # Add missing columns if needed
            for col in expected_columns:
                if col not in dataset_df.columns:
                    dataset_df[col] = np.nan

    datasets.append(dataset_df)

# Combine all datasets (optional: add a dataset_id column)
final_df = pd.concat(datasets, ignore_index=True)

# Optional: Add source info
final_df['dataset_id'] = [i for i, d in enumerate(datasets) for _ in range(len(d))]

# Save reshaped data
output_path = "Eddie_reshaped.xlsx"
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    final_df.to_excel(writer, sheet_name="parameters_clean", index=False)

print(f"✅ Reshaped data saved to {output_path}")
