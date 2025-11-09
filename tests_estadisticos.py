import pandas as pd
from itertools import combinations, product
import os

# Load datasets
air = pd.read_csv(os.path.join('Datos_PowerBI', 'SISAIRE-MODIFIED-COMPLETE-FIXED-NOWCAST.csv'), sep=';', decimal=',')

# User-provided variable list for filter
air_variables = ['CO','NO2','O3','PM10','PM25','SO2','Nowcast_PM25','Nowcast_PM10']

# --- FIX IS HERE ---
# Convert all specified columns to a numeric type.
# Any values that cannot be converted will become NaN (Not a Number).
for col in air_variables:
    air[col] = pd.to_numeric(air[col], errors='coerce')

# You can uncomment the line below to verify the data types are now float64
# print(air.info())

localidades = sorted(list(set(air['Localidad'].astype(str).unique())))
years = sorted(list(set(air['Año'].dropna().unique()))) # Added dropna() for safety

results = []
def filter_and_stats(variable, localidad, year_start, year_end):
    df = air.copy()
    # Filter localidad
    if localidad != 'ALL':
        df = df[df['Localidad'] == localidad]
    # Filter year range
    df = df[(df['Año'] >= year_start) & (df['Año'] <= year_end)]
    # Calculate statistics
    n_rows = len(df)
    # The .mean() and .std() functions will now work correctly
    mean = df[variable].mean() if n_rows > 0 else None
    std = df[variable].std() if n_rows > 1 else None
    results.append({
        'variable': variable,
        'localidad': localidad,
        'year_start': year_start,
        'year_end': year_end,
        'mean': mean,
        'std': std,
        'n': n_rows
    })

# Demo loop (limit for testing -- can expand)
for variable in air_variables:
    for localidad in ['ALL'] + localidades:
        for year_start in years:
            for year_end in years:
                if year_end < year_start:
                    continue
                filter_and_stats(variable, localidad, year_start, year_end)

results_df = pd.DataFrame(results)
results_df.to_csv('statistical_results_for_powerbi.csv', index=False)
print('Statistical results table ready for Power BI slicers!')