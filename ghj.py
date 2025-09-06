#!/usr/bin/env python3
"""
Reshape Eddie.xlsx: Aligns multiple datasets in the 'parameters' sheet.
The second dataset (often transposed or misaligned) is automatically reshaped
to match the structure of the first dataset.

Handles:
- Sheet name variations (case, spaces)
- Dataset separation by empty rows
- Transposed data detection and correction
- Duplicate/NaN column name cleanup
- Final concatenation and export

Author: Your Name
License: MIT
"""

import pandas as pd
import numpy as np
import os


def find_dataset_ranges(dataframe):
    """
    Detect dataset blocks separated by empty rows.
    Returns list of (start_row, end_row) tuples.
    """
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
    # Catch last dataset if sheet doesn't end with empty row
    if start is not None:
        ranges.append((start, len(dataframe)))
    return ranges


def make_columns_unique(columns):
    """
    Replace NaN, empty, or duplicate column names with 'Unnamed_i'.
    Ensures safe concatenation later.
    """
    seen = {}
    new_cols = []
    for col in columns:
        # Check if column name is invalid or duplicate
        if pd.isna(col) or str(col).strip() == "" or col in seen:
            i = 1
            while f"Unnamed_{i}" in seen:
                i += 1
            new_name = f"Unnamed_{i}"
            seen[new_name] = True
            new_cols.append(new_name)
        else:
            seen[col] = True
            new_cols.append(col)
    return new_cols


def main():
    # ==============================
    # CONFIGURATION
    # ==============================
    INPUT_FILE = "Eddie.xlsx"
    OUTPUT_FILE = "Eddie_reshaped.xlsx"
    TARGET_SHEET_NAME = "parameters"  # Case-insensitive, ignores surrounding spaces

    # ==============================
    # STEP 1: VALIDATE INPUT FILE
    # ==============================
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"❌ Input file '{INPUT_FILE}' not found in current directory.")

    # Load Excel file to inspect sheets
    xls = pd.ExcelFile(INPUT_FILE)
    print(f"📁 Available sheets: {xls.sheet_names}")

    # Auto-detect target sheet (flexible matching)
    target_sheet = None
    for sheet in xls.sheet_names:
        if sheet.strip().lower() == TARGET_SHEET_NAME.lower():
            target_sheet = sheet
            break

    if target_sheet is None:
        raise ValueError(f"❌ No sheet found matching '{TARGET_SHEET_NAME}' (case-insensitive, ignores spaces).")

    print(f"✅ Using sheet: '{target_sheet}'")

    # ==============================
    # STEP 2: LOAD DATA
    # ==============================
    df = pd.read_excel(INPUT_FILE, sheet_name=target_sheet, header=None)
    print(f"📊 Sheet shape: {df.shape}")

    # ==============================
    # STEP 3: SPLIT INTO DATASETS
    # ==============================
    ranges = find_dataset_ranges(df)
    print(f"🧩 Found {len(ranges)} datasets separated by empty rows.")

    # ==============================
    # STEP 4: PROCESS EACH DATASET
    # ==============================
    datasets = []

    for idx, (start, end) in enumerate(ranges):
        print(f"  → Processing dataset {idx + 1} (rows {start} to {end})")

        # Extract block and set first row as header
        raw_block = df.iloc[start:end].reset_index(drop=True)
        if len(raw_block) == 0:
            continue

        raw_block.columns = raw_block.iloc[0]  # Header row
        dataset_df = raw_block[1:].reset_index(drop=True)  # Data rows

        # Clean column names (replace NaN/duplicates)
        dataset_df.columns = make_columns_unique(dataset_df.columns)

        # Special handling for SECOND dataset (index 1)
        if idx == 1 and len(datasets) > 0:
            print("    ⚠️  Reshaping second dataset to match first...")

            expected_columns = datasets[0].columns.tolist()
            current_columns = dataset_df.columns.tolist()

            print(f"      Expected columns: {expected_columns}")
            print(f"      Current columns:   {current_columns}")

            # Heuristic: Transpose if structure doesn't match
            if (len(dataset_df) < len(dataset_df.columns)) or not any(
                str(col).strip().lower() in [str(e).strip().lower() for e in expected_columns]
                for col in current_columns
            ):
                print("      ↻ Transposing dataset...")
                transposed = dataset_df.T.reset_index(drop=True)
                transposed.columns = transposed.iloc[0]  # New header
                dataset_df = transposed[1:].reset_index(drop=True)
                dataset_df.columns = make_columns_unique(dataset_df.columns)
                print(f"      After transpose: {list(dataset_df.columns)}")

            # Align columns by position
            aligned_columns = [
                expected_columns[i] if i < len(expected_columns) else col
                for i, col in enumerate(dataset_df.columns)
            ]
            dataset_df.columns = aligned_columns

            # Add missing columns
            for col in expected_columns:
                if col not in dataset_df.columns:
                    dataset_df[col] = np.nan
                    print(f"      ➕ Added missing column: {col}")

            # Reorder to match first dataset
            dataset_df = dataset_df[expected_columns]

        # Final column cleanup before storing
        dataset_df.columns = make_columns_unique(dataset_df.columns)
        datasets.append(dataset_df)

    # ==============================
    # STEP 5: COMBINE & EXPORT
    # ==============================
    if not datasets:
        raise ValueError("❌ No datasets processed. Check input structure.")

    final_df = pd.concat(datasets, ignore_index=True)

    # Add metadata column
    final_df['source_dataset'] = [
        i + 1 for i, dataset in enumerate(datasets) for _ in range(len(dataset))
    ]

    # Export
    final_df.to_excel(OUTPUT_FILE, sheet_name="parameters_clean", index=False)

    print(f"\n🎉 SUCCESS!")
    print(f"✅ Output saved to: {OUTPUT_FILE}")
    print(f"📈 Final shape: {final_df.shape}")
    print(f"📋 Final columns: {list(final_df.columns)}")


if __name__ == "__main__":
    main()
