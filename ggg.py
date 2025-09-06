import pandas as pd
import os

file_path = "Eddie.xlsx"

if not os.path.exists(file_path):
    print("❌ File not found. Current directory:", os.getcwd())
    print("📁 Files here:", os.listdir())
else:
    print("✅ File found!")
    xls = pd.ExcelFile(file_path)
    print("📄 Sheets inside:", xls.sheet_names)
