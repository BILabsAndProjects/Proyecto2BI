import pandas as pd
import numpy as np
from scipy import stats
import warnings
import os

warnings.filterwarnings("ignore")

# Define los síntomas a analizar
SINTOMAS = {
    "Sibilancias": "Casos de sibilancia último año",
    "Tos": "Casos de tos último año diferente gripa",
    "Congestión nasal": "Casos de mocos o nariz tapada último año",
    "Todos": "Casos sintomas totales"
}

# Niveles de confianza para los cálculos
alphas = [0.1, 0.05, 0.01]
confidence_levels = [90, 95, 99]

# --- REUSED HELPER FUNCTIONS (No changes needed here) ---

def calculate_prevalence_symptom(df_group, symptom_col):
    """Calcula la prevalencia para un síntoma específico en un grupo de datos."""
    total_registros = df_group["Registros"].sum()
    total_casos = df_group[symptom_col].sum()

    if total_registros == 0:
        return 0, 0, 0

    prevalencia = total_casos / total_registros if total_registros > 0 else 0
    se = np.sqrt(prevalencia * (1 - prevalencia) / total_registros) if prevalencia > 0 and prevalencia < 1 else 0
    return prevalencia, se, total_registros

def calculate_pr_with_ci_and_pvalue(prev1, se1, n1, prev2, se2, n2, alpha):
    """Calcula PR, intervalos de confianza y p-valor."""
    if prev2 == 0 or n1 == 0 or n2 == 0:
        return np.nan, np.nan, np.nan, np.nan

    pr = prev1 / prev2
    if prev1 > 0:
        se_log_pr = np.sqrt(((1 - prev1) / (n1 * prev1)) + ((1 - prev2) / (n2 * prev2)))
        z = stats.norm.ppf(1 - alpha / 2)
        log_pr = np.log(pr)
        ci_lower = np.exp(log_pr - z * se_log_pr)
        ci_upper = np.exp(log_pr + z * se_log_pr)
        z_stat = log_pr / se_log_pr
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    else:
        ci_lower, ci_upper, p_value = 0, np.inf, 1.0
    return pr, ci_lower, ci_upper, p_value

# --- MAIN ANALYSIS SCRIPT ---

def main():
    print("=" * 70)
    print("ANÁLISIS DE PR: PREVALENCIA POR LOCALIDAD VS PROMEDIO DE BOGOTÁ")
    print("=" * 70)

    # Cargar datos de salud
    try:
        file_path_health = os.path.join('Datos_PowerBI', 'obs_aire_menores5anos_procesado_fixed.csv')
        df_health = pd.read_csv(file_path_health, sep=";", decimal=',')
        print(f"Datos de salud cargados: {len(df_health)} filas.")
    except FileNotFoundError:
        print(f"\nError: No se encontró el archivo: {file_path_health}")
        return

    resultados = []
    lista_localidades = sorted(df_health['Localidad'].unique())

    # Iterar sobre cada síntoma
    for sintoma_name, sintoma_col in SINTOMAS.items():
        print(f"\n--- Analizando Síntoma: {sintoma_name} ---")

        # 1. Calcular la prevalencia para toda Bogotá (nuestro grupo de referencia)
        prev_bogota, se_bogota, n_bogota = calculate_prevalence_symptom(df_health, sintoma_col)

        if n_bogota == 0 or prev_bogota == 0:
            print(f"No hay datos o la prevalencia es cero para '{sintoma_name}' en Bogotá. Omitiendo.")
            continue
        
        print(f"Prevalencia en Bogotá: {prev_bogota:.4f}")

        # 2. Iterar sobre cada localidad para compararla con el promedio de Bogotá
        for localidad in lista_localidades:
            df_localidad = df_health[df_health["Localidad"] == localidad]

            # Calcular prevalencia para la localidad específica (nuestro grupo de exposición)
            prev_localidad, se_localidad, n_localidad = calculate_prevalence_symptom(df_localidad, sintoma_col)
            
            if n_localidad == 0:
                continue # Omitir si la localidad no tiene datos

            # 3. Calcular el PR para cada nivel de confianza
            for alpha, conf_level in zip(alphas, confidence_levels):
                pr, ci_lower, ci_upper, p_value = calculate_pr_with_ci_and_pvalue(
                    prev_localidad, se_localidad, n_localidad,  # Grupo 1: La localidad
                    prev_bogota, se_bogota, n_bogota,            # Grupo 2: Bogotá
                    alpha
                )

                if not np.isnan(pr):
                    resultado = {
                        "Localidad": localidad,
                        "PR": pr,
                        "nivel_confianza": f"{conf_level}%",
                        "ic_inferior": ci_lower,
                        "ic_superior": ci_upper,
                        "p_valor": p_value,
                        "sintoma": sintoma_name
                    }
                    resultados.append(resultado)

    # 4. Guardar los resultados en un archivo CSV
    if resultados:
        df_resultados = pd.DataFrame(resultados)
        
        # Ordenar y redondear
        column_order = ["Localidad", "PR", "nivel_confianza", "ic_inferior", "ic_superior", "p_valor", "sintoma"]
        df_resultados = df_resultados[column_order]
        df_resultados['PR'] = df_resultados['PR'].round(2)
        df_resultados['ic_inferior'] = df_resultados['ic_inferior'].round(2)
        df_resultados['ic_superior'] = df_resultados['ic_superior'].round(2)
        df_resultados['p_valor'] = df_resultados['p_valor'].round(4)

        # Guardar a CSV
        output_file = os.path.join("Datos_PowerBI", "PR-LOCALIDADES.csv")
        df_resultados.to_csv(output_file, sep=";", index=False)

        print("\n" + "=" * 70)
        print(f"¡Análisis completo! Resultados guardados en: {output_file}")
        print("=" * 70)
        print(df_resultados.head(10).to_string(index=False))
    else:
        print("\nNo se pudieron generar resultados. Revisa los datos de entrada.")

if __name__ == "__main__":
    main()