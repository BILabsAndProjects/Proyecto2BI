import pandas as pd
import re

# Read the CSV file (using semicolon as delimiter)
df = pd.read_csv('Datos_PowerBI/regression_results_copy.csv', sep=';', decimal=',')

# Function to parse the Variable column
def parse_variable(variable):
    """
    - C(Localidad)[T.BOSA] -> tipo: Localidad, nombre: BOSA
    - C(Estrato)[T.2] -> tipo: Estrato, nombre: 2
    - Intercept -> tipo: Intercept, nombre: Intercept
    - PM25 -> tipo: Contaminante, nombre: PM25
    """
    # Pattern for C(Category)[T.Value]
    pattern = r'C\(([^)]+)\)\[T\.([^\]]+)\]'
    match = re.match(pattern, variable)
    
    if match:
        tipo = match.group(1)  # Extract category (e.g., "Localidad", "Estrato")
        nombre = match.group(2)  # Extract value (e.g., "BOSA", "2")
        return tipo, nombre
    elif variable == 'Intercept':
        return 'Intercept', 'Intercept'
    else:
        # For pollutants like PM25, CO, SO2, O3
        return 'Contaminante', variable

# Apply the parsing function
df[['tipo_variable', 'Variable_parsed']] = df['Variable'].apply(
    lambda x: pd.Series(parse_variable(x))
)

# Replace the original Variable column with the parsed name
df['Variable'] = df['Variable_parsed']

# Drop the temporary column
df = df.drop('Variable_parsed', axis=1)

# Reorder columns to have tipo_variable after Variable
cols = df.columns.tolist()
variable_idx = cols.index('Variable')
cols.insert(variable_idx + 1, cols.pop(cols.index('tipo_variable')))
df = df[cols]

# Save the parsed data to a new CSV file
output_file = 'Datos_PowerBI/regression_results_parsed.csv'
df.to_csv(output_file, sep=';', index=False, decimal='.')

print(f"Parsed data saved to: {output_file}")
print("\nFirst few rows:")
print(df.head(10))
print("\nUnique variable types:")
print(df['tipo_variable'].value_counts())
