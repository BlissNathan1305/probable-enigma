import pandas as pd
import numpy as np
import os

# ==============================
# STEP 1: LOAD FILE & AUTO-DETECT SHEET
# ==============================
file_path = "Eddie.xlsx"

# Check if file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"❌ Excel file not found: {file_path}")

# Load Excel and auto-find sheet matching 'parameters' (case-insensitive, ignores spaces)
xls = pd.ExcelFile(file_path)
target_sheet = None
for sheet in xls.sheet_names:
    if sheet.strip().lower() == "parameters":
        target_sheet = sheet
        break

if target_sheet is None:
    raise ValueError("❌ No sheet found matching 'parameters' (ignoring case and spaces)")

print(f"✅ Found sheet: '{target_sheet}'")

# ==============================
# STEP 2: LOAD SHEET WITHOUT HEADER (we'll detect headers per dataset)
# ==============================
df = pd.read_excel(file_path, sheet_name=target_sheet, header=None)

# ==============================
# STEP 3: SPLIT INTO DATASETS (separated by empty rows)
# ==============================
def find_dataset_ranges(dataframe):
    ranges = []
    start = None
    for i in range(len(dataframe)):
        if dataframe.iloc[i].notna().any():  # Non-empty row
            if start is None:
                start = i
        else:  # Empty row
            if start is not None:
                ranges.append((start, i))
                start = None
    # Catch last dataset if not followed by empty row
    if start is not None:
        ranges.append((start, len(dataframe)))
    return ranges

ranges = find_dataset_ranges(df)
print(f"📊 Found {len(ranges)} datasets in sheet.")

# ==============================
# STEP 4: PROCESS EACH DATASET
# ==============================
datasets = []

for idx, (start, end) in enumerate(ranges):
    print(f"  → Processing dataset {idx + 1} (rows {start} to {end})")
    
    # Extract raw block
    raw_block = df.iloc[start:end].reset_index(drop=True)
    
    # Use first row as header
    if len(raw_block) == 0:
        continue
    
    raw_block.columns = raw_block.iloc[0]  # Set header
    dataset_df = raw_block[1:].reset_index(drop=True)  # Remove header row from data
    
    # For SECOND dataset (index 1), reshape to match first dataset
    if idx == 1:
        print("    ⚠️  Reshaping second dataset...")
        
        # Get expected columns from first dataset
        if len(datasets) > 0:
            expected_columns = datasets[0].columns.tolist()
            current_columns = dataset_df.columns.tolist()
            
            print(f"      Expected columns: {expected_columns}")
            print(f"      Current columns:   {current_columns}")
            
            # OPTION 1: If transposed (more columns than rows, or column names look like data)
            if len(dataset_df) < len(dataset_df.columns) or not any(col in expected_columns for col in current_columns):
                print("      ↻ Transposing dataset...")
                dataset_df = dataset_df.T.reset_index()
                dataset_df.columns = dataset_df.iloc[0]  # First row becomes header again
                dataset_df = dataset_df[1:].reset_index(drop=True)
                current_columns = dataset_df.columns.tolist()
                print(f"      After transpose columns: {current_columns}")
            
            # OPTION 2: Align column names — rename or add missing ones
            new_columns = []
            for i, col in enumerate(current_columns):
                if i < len(expected_columns):
                    new_columns.append(expected_columns[i])
                else:
                    new_columns.append(col)  # Keep extra columns if any
            
            dataset_df.columns = new_columns
            
            # Add missing expected columns
            for col in expected_columns:
                if col not in dataset_df.columns:
                    dataset_df[col] = np.nan
                    print(f"      ➕ Added missing column: {col}")
            
            # Reorder to match first dataset
            dataset_df = dataset_df[expected_columns]
    
    datasets.append(dataset_df)

# ==============================
# STEP 5: COMBINE ALL DATASETS
# ==============================
if len(datasets) == 0:
    raise ValueError("No datasets found to process.")

final_df = pd.concat(datasets, ignore_index=True)

# Optional: Add dataset source column
final_df['source_dataset'] = [i+1 for i, d in enumerate(datasets) for _ in range(len(d))]

# ==============================
# STEP 6: SAVE TO NEW FILE
# ==============================
output_path = "Eddie_reshaped.xlsx"
final_df.to_excel(output_path, sheet_name="parameters_clean", index=False)

print(f"\n🎉 SUCCESS!")
print(f"✅ Reshaped data saved to: {output_path}")
print(f"📈 Final shape: {final_df.shape}")
print(f"📋 Columns: {list(final_df.columns)}")
