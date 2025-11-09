import pandas as pd
import numpy as np
import os
from scipy import stats
from tqdm import tqdm
from itertools import product

# --- 1. Load and Prepare Data ---

# Load the dataset
file_path = os.path.join('Datos_PowerBI', 'SISAIRE-MODIFIED-COMPLETE-IMPUTED-FIXED-NOWCAST.csv')
air = pd.read_csv(file_path, sep=';', decimal=',')

# Define variables and confidence levels (as alpha values)
air_variables = ['CO', 'NO2', 'O3', 'PM10', 'PM25', 'SO2', 'Nowcast_PM25', 'Nowcast_PM10']
alphas = [0.05, 0.10, 0.01] # For 95%, 90%, and 99% confidence

# Convert numeric columns, coercing errors to NaN
for col in air_variables:
    air[col] = pd.to_numeric(air[col], errors='coerce')

# Get unique values for loops
localidades = sorted(list(set(air['Localidad'].astype(str).unique())))
years = sorted(list(set(air['Año'].dropna().astype(int).unique())))

# --- 2. Define the Calculation Function (Modified) ---

def calculate_confidence_intervals_long(data_series, alpha_levels):
    """
    Calculates stats and returns a LIST of dictionaries, one for each confidence level.
    """
    data = data_series.dropna()
    n = len(data)

    if n < 2:
        return [] # Return an empty list if calculation is not possible

    mean = data.mean()
    std = data.std()
    se = std / np.sqrt(n)

    # This list will hold a result for each confidence level
    results_list = []
    
    for alpha in alpha_levels:
        t_critical = stats.t.ppf(1 - alpha / 2, df=n - 1)
        margin_of_error = t_critical * se
        confidence_level_str = f"{round((1 - alpha) * 100)}%"
        
        # Create a dictionary for this specific confidence level
        results_list.append({
            'mean': mean,
            'std': std,
            'n': n,
            'confidence_level': confidence_level_str,
            'ci_lower': mean - margin_of_error,
            'ci_upper': mean + margin_of_error
        })
        
    return results_list

# --- 3. Generate All Combinations and Run Calculations ---

year_ranges = []
for start in years:
    for end in years:
        if end >= start:
            year_ranges.append((start, end))

all_combinations = list(product(air_variables, ['ALL'] + localidades, year_ranges))

final_results = []

# Loop through all combinations with a single tqdm progress bar
for combo in tqdm(all_combinations, desc="Calculating Statistics"):
    variable, localidad, (year_start, year_end) = combo
    
    # Filter the dataframe
    df_filtered = air.copy()
    if localidad != 'ALL':
        df_filtered = df_filtered[df_filtered['Localidad'] == localidad]
    df_filtered = df_filtered[(df_filtered['Año'] >= year_start) & (df_filtered['Año'] <= year_end)]

    # This now returns a list of results (e.g., for 90%, 95%, 99%)
    stats_list = calculate_confidence_intervals_long(df_filtered[variable], alphas)

    # Add the common filter info to each result and append
    if stats_list:
        for stats_dict in stats_list:
            record = {
                'variable': variable,
                'localidad': localidad,
                'year_start': year_start,
                'year_end': year_end
            }
            record.update(stats_dict)
            final_results.append(record)

# --- 4. Save Results ---

results_df = pd.DataFrame(final_results)

output_file = 'IMPUTED-confidence_interval_results_for_powerbi_long.csv'
results_df.to_csv(os.path.join("Datos_PowerBI", output_file), index=False, sep=';', decimal=',')

print(f"\nProcessing complete! Results saved to '{output_file}'")
print("The file is ready to be loaded into Power BI.")
results_df.head(6) # Display more rows to show the new structure