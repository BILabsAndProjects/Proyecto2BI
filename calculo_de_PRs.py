import pandas as pd
import numpy as np
from scipy import stats
import warnings
import os

warnings.filterwarnings("ignore")

CONTAMINANTES = ['CO', 'NO2', 'O3', 'PM10', 'PM25', 'SO2', 'Nowcast_PM25', 'Nowcast_PM10']
SINTOMAS = {
        "Sibilancias": "Casos de sibilancia último año",
        "Tos": "Casos de tos último año diferente gripa",
        "Congestión nasal": "Casos de mocos o nariz tapada último año",
        "Todos": "Casos sintomas totales"
    }

# Niveles de confianza
alphas = [0.1, 0.05, 0.01]
confidence_levels = [90, 95, 99]

IBOCA_THRESHOLDS = {
    'Nowcast_PM25': 12.1,
    'PM25': 12.1, 
    'Nowcast_PM10': 27.3,
    'PM10': 27.3,
    'O3': 73,
    'NO2': 28.6,
    'SO2': 9.7,
    'CO': 2550
}

def load_air_quality_data(file_path):
    """
    Carga y procesa los datos de calidad del aire
    """
    print("Cargando datos de calidad del aire...")
    df_air = pd.read_csv(file_path, sep=';', decimal='.')
    
    # Convertir fecha a datetime
    df_air['Fecha'] = pd.to_datetime(df_air['Fecha'])
    
    # Asegurar que Localidad esté en mayúsculas y sin tildes para matching
    df_air['Localidad'] = df_air['Localidad'].str.upper()
    df_air['Localidad'] = df_air['Localidad'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    
    
    # Agrupar por localidad (promedio de todos los años)
    df_air_promedio = df_air.groupby(['Localidad'])[CONTAMINANTES].mean().reset_index()
    
    print(f"Datos de calidad del aire cargados: {len(df_air_promedio)} localidades")
    print(f"Localidades disponibles: {sorted(df_air_promedio['Localidad'].unique())}")
    
    return df_air_promedio


def classify_localities_by_pollutant(df_air_promedio, pollutant='PM25'):
    """
    Clasifica localidades en alta/baja concentración para un contaminante específico
    """
    # Calcular mediana del contaminante
    median_value = df_air_promedio[pollutant].median()
    
    # Clasificar localidades
    localidades_alta = df_air_promedio[df_air_promedio[pollutant] >= median_value]['Localidad'].tolist()
    localidades_baja = df_air_promedio[df_air_promedio[pollutant] < median_value]['Localidad'].tolist()
    
    return localidades_alta, localidades_baja, median_value

def classify_localities_by_iboca(df_air_promedio, pollutant, thresholds):
    """
    Clasifica localidades usando umbrales fijos del IBOCA.
    - 'Bajo': Concentración <= umbral Verde.
    - 'Moderado o Superior': Concentración > umbral Verde.
    """
    if pollutant not in thresholds:
        print(f"Advertencia: El contaminante '{pollutant}' no tiene un umbral IBOCA definido. Se omitirá.")
        return [], []

    # Obtener el umbral para el nivel Verde (Bajo)
    umbral_verde = thresholds[pollutant]
    
    # Clasificar localidades
    # Grupo de referencia (Bajo riesgo)
    localidades_baja = df_air_promedio[df_air_promedio[pollutant] <= umbral_verde]['Localidad'].tolist()
    
    # Grupo de exposición (Riesgo moderado o superior)
    localidades_alta = df_air_promedio[df_air_promedio[pollutant] > umbral_verde]['Localidad'].tolist()
    
    print(f"Clasificación para {pollutant} (Umbral IBOCA: {umbral_verde} µg/m³):")
    print(f"  - Localidades de Bajo Riesgo ({len(localidades_baja)}): {localidades_baja}")
    print(f"  - Localidades de Riesgo Moderado o Superior ({len(localidades_alta)}): {localidades_alta}")
    
    return localidades_baja, localidades_alta


def calculate_prevalence(df_group):
    """
    Calcula la prevalencia para un grupo de datos
    """
    total_registros = df_group["Registros"].sum()
    total_casos = df_group["Casos sintomas totales"].sum()

    if total_registros == 0:
        return 0, 0, 0

    prevalencia = total_casos / total_registros
    se = np.sqrt(prevalencia * (1 - prevalencia) / total_registros)

    return prevalencia, se, total_registros


def calculate_prevalence_symptom(df_group, symptom_col):
    """
    Calcula la prevalencia para un síntoma específico
    """
    total_registros = df_group["Registros"].sum()
    total_casos = df_group[symptom_col].sum()

    if total_registros == 0:
        return 0, 0, 0

    prevalencia = total_casos / total_registros
    se = np.sqrt(prevalencia * (1 - prevalencia) / total_registros) if prevalencia > 0 and prevalencia < 1 else 0

    return prevalencia, se, total_registros


def calculate_pr_with_ci_and_pvalue(prev1, se1, n1, prev2, se2, n2, alpha):
    """
    Calcula Prevalence Ratio, intervalos de confianza y p-valor
    """
    if prev2 == 0 or n1 == 0 or n2 == 0:
        return np.nan, np.nan, np.nan, np.nan

    # Prevalence Ratio
    pr = prev1 / prev2

    # Error estándar del log(PR)
    if prev1 > 0:
        se_log_pr = np.sqrt((1 - prev1) / (n1 * prev1) + (1 - prev2) / (n2 * prev2))
        
        # Z-score para el nivel de confianza
        z = stats.norm.ppf(1 - alpha / 2)
        
        # IC en escala log
        log_pr = np.log(pr)
        ci_lower_log = log_pr - z * se_log_pr
        ci_upper_log = log_pr + z * se_log_pr
        
        # Transformar de vuelta a escala original
        ci_lower = np.exp(ci_lower_log)
        ci_upper = np.exp(ci_upper_log)
        
        # Calcular p-valor (test de Wald)
        z_stat = log_pr / se_log_pr
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    else:
        ci_lower = 0
        ci_upper = np.inf
        p_value = 1.0

    return pr, ci_lower, ci_upper, p_value

def pollutant_concentration_analysis(df_air, df_health, pollutant, resultados):
    loc_alta_contaminante, loc_baja_contaminante = classify_localities_by_iboca(df_air, pollutant, IBOCA_THRESHOLDS)
    
    if len(loc_alta_contaminante) > 0 and len(loc_baja_contaminante) > 0:
        print(loc_alta_contaminante)
        for sintoma_name, sintoma_col in SINTOMAS.items():
            df_health_alta = df_health[df_health["Localidad"].isin(loc_alta_contaminante)]
            df_health_baja = df_health[df_health["Localidad"].isin(loc_baja_contaminante)]
            
            if sintoma_name == "Todos":
                prev_alta, se_alta, n_alta = calculate_prevalence(df_health_alta)
                prev_baja, se_baja, n_baja = calculate_prevalence(df_health_baja)
            else:
                prev_alta, se_alta, n_alta = calculate_prevalence_symptom(df_health_alta, sintoma_col)
                prev_baja, se_baja, n_baja = calculate_prevalence_symptom(df_health_baja, sintoma_col)
                
            if n_alta > 0 and n_baja > 0 and prev_baja > 0:
                for alpha, conf_level, in zip(alphas, confidence_levels):
                    pr, ci_lower, ci_upper, p_value = calculate_pr_with_ci_and_pvalue(
                        prev_alta, se_alta, n_alta, prev_baja, se_baja, n_baja, alpha
                    )
                    
                    if not np.isnan(pr):
                        if "25" in pollutant:
                            pollutant = pollutant.replace("25", "2.5")
                        result = {
                            "comparación": f"{pollutant} Alto vs Bajo",
                            "PR": pr,
                            "nivel_confianza": f"{conf_level}%",
                            "ic_inferior": ci_lower,
                            "ic_superior": ci_upper,
                            "p_valor": p_value,
                            "sintoma": sintoma_name
                        }
                        resultados.append(result)
    return resultados
                
                
def main():
    # Cargar datos de salud
    print("Cargando datos de salud...")
    file_path_health = os.path.join('Datos_PowerBI', 'obs_aire_menores5anos_procesado_fixed.csv')
    df_health = pd.read_csv(file_path_health, sep=";", decimal=',')
    
    # Cargar datos de calidad del aire
    file_path_air = os.path.join('Datos_PowerBI', 'SISAIRE-MODIFIED-COMPLETE-IMPUTED-FIXED-NOWCAST.csv')
    df_air = load_air_quality_data(file_path_air)

    # Lista para almacenar resultados
    resultados = []

    # ==========================================================================
    # 1. ANÁLISIS POR ESTRATO (Bajo vs Alto)
    # ==========================================================================
    print("\n1. PR por Estrato Socioeconómico...")
    
    for sintoma_name, sintoma_col in SINTOMAS.items():
        # Filtrar estratos
        df_bajo = df_health[df_health["Estrato"].isin([1, 2])]
        df_alto = df_health[df_health["Estrato"].isin([5, 6])]
        
        if sintoma_name == "Todos":
            prev_bajo, se_bajo, n_bajo = calculate_prevalence(df_bajo)
            prev_alto, se_alto, n_alto = calculate_prevalence(df_alto)
        else:
            prev_bajo, se_bajo, n_bajo = calculate_prevalence_symptom(df_bajo, sintoma_col)
            prev_alto, se_alto, n_alto = calculate_prevalence_symptom(df_alto, sintoma_col)
        
        if n_bajo > 0 and n_alto > 0 and prev_alto > 0:
            for alpha, conf_level in zip(alphas, confidence_levels):
                pr, ci_lower, ci_upper, p_value = calculate_pr_with_ci_and_pvalue(
                    prev_bajo, se_bajo, n_bajo, prev_alto, se_alto, n_alto, alpha
                )
                
                if not np.isnan(pr):
                    resultado = {
                        "comparación": "Estratos 1-2 vs 5-6",
                        "PR": pr,
                        "nivel_confianza": f"{conf_level}%",
                        "ic_inferior": ci_lower,
                        "ic_superior": ci_upper,
                        "p_valor": p_value,
                        "sintoma": sintoma_name
                    }
                    resultados.append(resultado)

    # ==========================================================================
    # 2. ANÁLISIS POR CONCENTRACIÓN DE CONTAMINANTES
    # ==========================================================================
    print("2. PR por Concentración de Contaminantes...")
    
    for pollutant in CONTAMINANTES:
        resultados = pollutant_concentration_analysis(df_air, df_health, pollutant, resultados)
    
    # ==========================================================================
    # GUARDAR RESULTADOS EN CSV
    # ==========================================================================
    df_resultados = pd.DataFrame(resultados)

    if len(df_resultados) > 0:
        # Ordenar columnas según el formato solicitado
        columnas_orden = [
            "comparación",
            "PR",
            "nivel_confianza",
            "ic_inferior",
            "ic_superior",
            "p_valor",
            "sintoma"
        ]
        df_resultados = df_resultados[columnas_orden]
        
        # Redondear valores numéricos
        df_resultados['PR'] = df_resultados['PR'].round(4)
        df_resultados['ic_inferior'] = df_resultados['ic_inferior'].round(4)
        df_resultados['ic_superior'] = df_resultados['ic_superior'].round(4)
        df_resultados['p_valor'] = df_resultados['p_valor'].round(6)
        
        # Guardar a CSV
        archivo_salida = os.path.join("Datos_PowerBI", "PR-IMPUTED.csv")
        df_resultados.to_csv(archivo_salida, sep=";", index=False)

        print("\n" + "=" * 70)
        print(f"RESULTADOS GUARDADOS EN: {archivo_salida}")
        print("=" * 70)

        # Mostrar resumen
        print("\nRESUMEN DE RESULTADOS:")
        print(f"Total de PR calculados: {len(df_resultados)}")
        
        # Mostrar estadísticas por tipo de comparación
        print("\n--- Resumen por Tipo de Comparación ---")
        comparaciones_unicas = df_resultados['comparación'].unique()
        for comp in comparaciones_unicas:
            df_comp = df_resultados[df_resultados['comparación'] == comp]
            print(f"\n{comp}:")
            
            # Mostrar para cada síntoma con IC 95%
            df_95 = df_comp[df_comp['nivel_confianza'] == '95%']
            for _, row in df_95.iterrows():
                print(f"  {row['sintoma']}: PR={row['PR']:.4f} (IC95%: {row['ic_inferior']:.4f}-{row['ic_superior']:.4f}), p={row['p_valor']:.4f}")
        
        # Mostrar resultados significativos (p < 0.05)
        print("\n--- Resultados Significativos (p < 0.05) ---")
        df_sig = df_resultados[(df_resultados['p_valor'] < 0.05) & (df_resultados['nivel_confianza'] == '95%')]
        if len(df_sig) > 0:
            for _, row in df_sig.iterrows():
                print(f"{row['comparación']} - {row['sintoma']}: PR={row['PR']:.4f}, p={row['p_valor']:.4f}")
        else:
            print("No se encontraron asociaciones significativas con p < 0.05")
        
        # Tabla resumen
        print("\n--- Primeras 15 filas del archivo de salida ---")
        print(df_resultados.head(15).to_string(index=False))
        
        # Estadísticas generales
        print("\n--- Estadísticas Generales ---")
        print(f"Comparaciones únicas: {len(comparaciones_unicas)}")
        print(f"Síntomas analizados: {', '.join(df_resultados['sintoma'].unique())}")
        print(f"Niveles de confianza: {', '.join(df_resultados['nivel_confianza'].unique())}")
        
        # Identificar PR más altos
        print("\n--- Top 5 PR más altos (IC 95%) ---")
        df_95 = df_resultados[df_resultados['nivel_confianza'] == '95%']
        df_top = df_95.nlargest(5, 'PR')[['comparación', 'sintoma', 'PR', 'p_valor']]
        for _, row in df_top.iterrows():
            print(f"{row['comparación']} - {row['sintoma']}: PR={row['PR']:.4f}, p={row['p_valor']:.4f}")
            
    else:
        print("\nNo se pudieron calcular prevalence ratios con los datos proporcionados.")
        print("Verifica que:")
        print("  1. Existan casos en los estratos 1, 2, 5 y 6")
        print("  2. Las localidades en el archivo de salud coincidan con las del archivo de aire")
        print("  3. Los archivos CSV estén en la carpeta 'Datos_PowerBI'")


if __name__ == "__main__":
    print("=" * 70)
    print("ANÁLISIS DE PREVALENCE RATIOS")
    print("Comparaciones a través de todo el período de datos")
    print("=" * 70)
    
    try:
        # Ejecutar análisis principal
        main()
    except FileNotFoundError as e:
        print(f"\nError: No se encontró el archivo: {e}")
        print("Asegúrate de que los siguientes archivos estén en la carpeta 'Datos_PowerBI':")
        print("  - obs_aire_menores5anos_procesado_fixed.csv")
        print("  - SISAIRE-MODIFIED-COMPLETE-IMPUTED-FIXED-NOWCAST.csv")
    except Exception as e:
        print(f"\nError inesperado: {e}")
        import traceback
        traceback.print_exc()