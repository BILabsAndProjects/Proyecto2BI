import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
import os

warnings.filterwarnings("ignore")

CONTAMINANTES = ['CO', 'NO2', 'O3', 'SO2', 'PM25', 'PM10' ] # 'Nowcast_PM25', 'Nowcast_PM10'
SINTOMAS = {
    "Sibilancias": "Casos de sibilancia último año",
    "Tos": "Casos de tos último año diferente gripa",
    "Congestión_nasal": "Casos de mocos o nariz tapada último año",
    "Todos": "Casos sintomas totales"
}


def prepare_data_for_modeling(health_path, air_path):
    """
    Prepara el dataset final para el análisis de regresión y correlación.
    """
    print("1. Preparando los datos para el modelado...")
    
    df_health = pd.read_csv(health_path, sep=";", decimal=',')
    
    # Cargar y pre-procesar datos de aire
    df_air = pd.read_csv(air_path, sep=';', decimal='.')
    df_air['Fecha'] = pd.to_datetime(df_air['Fecha'])
    df_air['Año'] = df_air['Fecha'].dt.year
    df_air['Localidad'] = df_air['Localidad'].str.upper().str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    df_air_avg = df_air.groupby(['Año', 'Localidad'])[CONTAMINANTES].mean().reset_index()

    # Agregar datos de salud para juntar Sexo ya que no se toma en cuenta en el modelo
    grouping_cols = ['Año', 'Localidad', 'Estrato'] # tampoco estamos incluyendo aseguramiento
    cols_to_sum = list(SINTOMAS.values()) + ['Registros']
    df_health_agg = df_health.groupby(grouping_cols)[cols_to_sum].sum().reset_index()

    # Unir datos de salud agregados con los de calidad del aire
    df_model = pd.merge(df_health_agg, df_air_avg, on=['Año', 'Localidad'], how='left')

    # Calcular prevalencias (variables dependientes)
    for name, col in SINTOMAS.items():
        prevalence_col = f'prevalencia_{name}'
        df_model[prevalence_col] = (df_model[col] / df_model['Registros']).fillna(0)
    
    scaler = StandardScaler()
    # solo estandarizar columnas de contamiantes
    contaminantes = CONTAMINANTES
    
    cols_to_scale = [c for c in contaminantes if c in df_model.columns]
    df_model[cols_to_scale] = scaler.fit_transform(df_model[cols_to_scale])
    print("Variables de contaminantes estandarizadas (media 0, std 1).")
    
    print(f"Datos preparados. Total de filas para el modelo: {len(df_model)}")
    return df_model

def run_correlation_analysis(df_model):
    """
    Calcula la matriz de correlación de Spearman y la guarda en un CSV.
    """
    print("\n2. Ejecutando análisis de correlación...")
    
    # Seleccionar variables numéricas relevantes
    cols_for_corr = [c for c in CONTAMINANTES]
    cols_for_corr += [f'prevalencia_{name}' for name in SINTOMAS.keys()]
    
    # Calcular matriz de correlación de Spearman
    corr_matrix = df_model[cols_for_corr].corr(method='pearson')
    
    output_file = os.path.join("Datos_PowerBI", "spearman_correlation.csv")
    corr_matrix.to_csv(output_file, sep=';', decimal=',')
    
    print(f"Matriz de correlación guardada en: {output_file}")

def run_regression_analysis(df_model):
    """
    Ejecuta modelos de regresión múltiple para cada síntoma y guarda los resultados.
    """
    print("\n3. Ejecutando análisis de regresión múltiple...")
    
    all_results = []
    
    # Variables predictoras
    predictors = [c for c in CONTAMINANTES]
    predictors += ['C(Estrato)', 'C(Localidad)']
    
    for sintoma_name in SINTOMAS.keys():
        print(f"  - Modelo para predecir: prevalencia_{sintoma_name}")
        
        y_var = f'prevalencia_{sintoma_name}'
        
        # Construir la fórmula para statsmodels
        formula = f"{y_var} ~ {' + '.join(predictors)}"
        
        model_cols = [y_var] + [c for c in CONTAMINANTES] + ['Estrato', 'Localidad']
        df_subset = df_model[model_cols].dropna()

        if len(df_subset) < len(predictors) + 1:
            print(f"    -> No hay suficientes datos para el modelo de {sintoma_name}. Omitiendo.")
            continue
            
        # Ajustar el modelo
        model = smf.ols(formula, data=df_subset).fit()
        
        results_summary = model.summary2().tables[1]
        results_summary = results_summary.reset_index().rename(columns={
            'index': 'Variable', 'Coef.': 'Coeficiente', 'Std.Err.': 'Error_estandar',
            'P>|t|': 'p_valor', '[0.025': 'IC_Inferior', '0.975]': 'IC_Superior'
        })
        
        # Añadir métricas globales del modelo
        results_summary['Sintoma_predicho'] = sintoma_name
        results_summary['R2_ajustado'] = model.rsquared_adj
        
        # Calcular RMSE
        predictions = model.predict(df_subset)
        rmse = np.sqrt(mean_squared_error(df_subset[y_var], predictions))
        results_summary['RMSE'] = rmse
        
        all_results.append(results_summary)

    if all_results:
        final_results_df = pd.concat(all_results, ignore_index=True)
        
        # Ordenar columnas
        column_order = ['Sintoma_predicho', 'Variable', 'Coeficiente', 'Error_estandar', 'p_valor', 
                        'IC_Inferior', 'IC_Superior', 'R2_ajustado', 'RMSE']
        final_results_df = final_results_df[column_order]

        output_file = os.path.join("Datos_PowerBI", "regression_results.csv")
        final_results_df.to_csv(output_file, sep=';', decimal=',', index=False)
        print(f"\nResultados de la regresión guardados en: {output_file}")
    else:
        print("\nNo se pudieron generar resultados de regresión.")


if __name__ == "__main__":
    health_data_path = os.path.join('Datos_PowerBI', 'obs_aire_menores5anos_procesado_fixed.csv')
    air_data_path = os.path.join('Datos_PowerBI', 'SISAIRE-MODIFIED-COMPLETE-IMPUTED-FIXED-NOWCAST.csv')
    
    # 1. Preparar el dataset unificado
    df_modelo_final = prepare_data_for_modeling(health_data_path, air_data_path)
    
    # 2. Correr análisis de correlación
    run_correlation_analysis(df_modelo_final)
    
    # 3. Correr análisis de regresión
    run_regression_analysis(df_modelo_final)
    
    print("\n¡Análisis completo!")
