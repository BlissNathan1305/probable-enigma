# ----------------------------------------------------------
# Cross-tab ➜ Tidy  (Edata.xlsx style)
# ----------------------------------------------------------
import pandas as pd
import numpy as np

# 1. Read the raw sheet -------------------------------------------------------
#    - no header yet (we will build it manually)
#    - keep the first column that contains month names
raw = pd.read_excel("Edata.xlsx", sheet_name="Sheet1", header=None)

# 2. Identify the row indices where each block starts -------------------------
block_starts = raw[raw[0].str.contains("Evaporation|Wind speed|UYO TEMP|UYO RAIN|STATION PRESSURE|RELATIVE HUMIDITY",
                                       na=False)].index.tolist()
block_starts.append(raw.shape[0])          # sentinel

# 3. Helper: turn one block into a tidy mini-data-frame -----------------------
def tidy_one_block(chunk, var_name):
    """
    chunk : DataFrame, 13 rows × 10 cols
            row 0  →  month/year,2011,2012,...,2019
            rows 1-12 → Jan..Dec data
    returns : DataFrame with columns [variable, year, month, value]
    """
    years = chunk.iloc[0, 1:].astype(int)          # 2011 … 2019
    months = chunk.iloc[1:, 0].str.strip()         # Jan … Dec
    values = chunk.iloc[1:, 1:].replace(",", np.nan).astype(float)

    df = (values.stack().reset_index())
    df.columns = ["month_idx", "year_idx", "value"]
    df["month"] = months.iloc[df.month_idx].values
    df["year"]  = years.iloc[df.year_idx].values
    df["variable"] = var_name
    return df[["variable", "year", "month", "value"]]

# 4. Loop over blocks and concatenate ----------------------------------------
tidy_frames = []
for i, start in enumerate(block_starts[:-1]):
    end = block_starts[i+1]
    chunk = raw.iloc[start:end, :12]             # keep only 12 data cols
    var_name = raw.iloc[start, 0].split(" ")[0]  # first word is enough
    tidy_frames.append(tidy_one_block(chunk, var_name))

tidy = pd.concat(tidy_frames, ignore_index=True)

# 5. Export & quick sanity check ---------------------------------------------
tidy.to_csv("tidy_Edata.csv", index=False)
print(tidy.head(30))

