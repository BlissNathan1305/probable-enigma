import pandas as pd
import numpy as np
import sys

def reshape_crosstab_to_tidy(file_path, sheet_name=0, id_vars=None, var_name='variable', value_name='value'):
    """
    Reshape cross-tab data into tidy format
    
    Parameters:
    file_path (str): Path to the Excel file
    sheet_name (str/int): Sheet name or index (default: 0)
    id_vars (list): Column(s) to use as identifier variables
    var_name (str): Name to use for the 'variable' column
    value_name (str): Name to use for the 'value' column
    
    Returns:
    pandas.DataFrame: Tidy format data
    """
    
    try:
        # Read the Excel file
        print(f"Reading file: {file_path}")
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        print("Original data shape:", df.shape)
        print("Original columns:", list(df.columns))
        print("\nFirst few rows of original data:")
        print(df.head())
        
        # If id_vars is not specified, try to identify them
        if id_vars is None:
            # Look for columns that are likely identifiers (non-numeric or first few columns)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
            
            if len(non_numeric_cols) > 0:
                id_vars = list(non_numeric_cols)
                print(f"Auto-detected identifier columns: {id_vars}")
            else:
                # Use the first column as identifier if no non-numeric columns found
                id_vars = [df.columns[0]]
                print(f"Using first column as identifier: {id_vars}")
        
        # Melt the dataframe to convert from wide to long format
        tidy_df = pd.melt(df, 
                         id_vars=id_vars, 
                         var_name=var_name, 
                         value_name=value_name)
        
        print(f"\nSuccessfully reshaped data!")
        print("Tidy data shape:", tidy_df.shape)
        print("\nFirst few rows of tidy data:")
        print(tidy_df.head())
        
        return tidy_df
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def save_tidy_data(tidy_df, output_file='tidy_data.xlsx'):
    """
    Save the tidy data to an Excel file
    
    Parameters:
    tidy_df (pandas.DataFrame): Tidy format data
    output_file (str): Output file name
    """
    try:
        tidy_df.to_excel(output_file, index=False)
        print(f"\nTidy data saved to: {output_file}")
    except Exception as e:
        print(f"Error saving file: {str(e)}")

def main():
    # File configuration
    input_file = 'Edata.xlsx'
    output_file = 'tidy_Edata.xlsx'
    
    # For more control, you can specify these parameters:
    # sheet_name = 0  # or 'Sheet1' or specific sheet name
    # id_vars = ['Region', 'Category']  # specify identifier columns
    # var_name = 'Metric'  # name for the variable column
    # value_name = 'Value'  # name for the value column
    
    print("=" * 60)
    print("CROSS-TAB TO TIDY DATA CONVERTER")
    print("=" * 60)
    
    # Reshape the data
    tidy_data = reshape_crosstab_to_tidy(
        file_path=input_file,
        sheet_name=0,  # Change if needed
        id_vars=None,  # Auto-detect or specify like ['Region', 'Category']
        var_name='Variable',
        value_name='Value'
    )
    
    if tidy_data is not None:
        # Save the results
        save_tidy_data(tidy_data, output_file)
        
        # Show some statistics
        print("\n" + "=" * 60)
        print("DATA SUMMARY:")
        print("=" * 60)
        print(f"Total rows in tidy data: {len(tidy_data)}")
        print(f"Unique variables: {tidy_data['Variable'].nunique()}")
        if 'Variable' in tidy_data.columns:
            print(f"Variables: {tidy_data['Variable'].unique()}")
        
        # Show data types
        print("\nData types:")
        print(tidy_data.dtypes)

if __name__ == "__main__":
    main()
