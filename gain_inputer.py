import pandas as pd
import numpy as np

from gain import gain

from sklearn.model_selection import ParameterGrid

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


dados = pd.read_csv(
    r"C:\Users\joaop\OneDrive\Área de Trabalho\TP1 Aprendizado de Máquina\dados_20mil.csv"
)

print("Início.")


colunas_nao_imputadas = [
    "ano",
    "id_municipio",
    "sigla_uf"
]

dados = dados.drop(
    columns=colunas_nao_imputadas,
    errors="ignore"
)

# converter para numérico
dados = dados.apply(
    pd.to_numeric,
    errors="coerce"
)



dados_original = dados.copy()



np.random.seed(42)

mask = (
    np.random.rand(*dados_original.shape) < 0.30
) & (~dados_original.isna())

dados_com_missing = dados_original.copy()

dados_com_missing[mask] = np.nan





dados_normalizados = dados_com_missing.values

def calcular_mape(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mask = np.abs(y_true) > 0.1

    return np.mean(
        np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
    ) * 100



param_grid = {
    "batch_size": [128],
    "hint_rate": [0.7],
    "alpha": [50],
    "iterations": [2000]
}

grid = list(ParameterGrid(param_grid))


resultados = []

for i, params in enumerate(grid):

    print(f"\nTestando combinação {i+1}/{len(grid)}")
    print(params)

    

    dados_imputados = gain(
        dados_normalizados,
        params
    )

    # voltar escala original
    dados_imputados = dados_imputados

    dados_imputados = pd.DataFrame(
        dados_imputados,
        columns=dados.columns
    )



    y_true = dados_original.values[mask]

    y_pred = dados_imputados.values[mask]

    # remover NaNs
    validos = (
        ~np.isnan(y_true)
    ) & (
        ~np.isnan(y_pred)
    )

    y_true = y_true[validos]

    y_pred = y_pred[validos]

    

    mae = mean_absolute_error(y_true, y_pred)

    mse = mean_squared_error(y_true, y_pred)

    rmse = np.sqrt(mse)

    mape = calcular_mape(y_true, y_pred)

    resultados.append({
        "batch_size": params["batch_size"],
        "hint_rate": params["hint_rate"],
        "alpha": params["alpha"],
        "iterations": params["iterations"],
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape
    })



resultados_df = pd.DataFrame(resultados)

resultados_df = resultados_df.sort_values(
    by="RMSE"
)

print("\n=========== MELHORES RESULTADOS ===========\n")

print(resultados_df.head(10))

# salvar grid
resultados_df.to_csv(
    r"C:\Users\joaop\OneDrive\Área de Trabalho\TP1 Aprendizado de Máquina\resultado_gain.csv",
    index=False
)



melhor = resultados_df.iloc[0]

print("\nMelhores hiperparâmetros:")
print(melhor)



parametros_finais = {
    "batch_size": int(melhor["batch_size"]),
    "hint_rate": melhor["hint_rate"],
    "alpha": melhor["alpha"],
    "iterations": int(melhor["iterations"])
}

# usar dados reais da base
dados_reais_normalizados = dados.values

dados_finais = gain(
    dados_reais_normalizados,
    parametros_finais
)


dados_finais = pd.DataFrame(
    dados_finais,
    columns=dados.columns
)



mascara_original = dados.isna()

for coluna in dados.columns:

    dados_finais.loc[
        ~mascara_original[coluna],
        coluna
    ] = dados.loc[
        ~mascara_original[coluna],
        coluna
    ]


dados_finais.to_csv(
    r"C:\Users\joaop\OneDrive\Área de Trabalho\TP1 Aprendizado de Máquina\base_imputada_gain.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nConcluído.")
